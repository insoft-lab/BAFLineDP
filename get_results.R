library(tidyverse)
library(gridExtra)
library(ModelMetrics)
library(caret)
library(reshape2)
library(pROC)
library(effsize)
library(progress)

# configure your absolute work path
setwd("your_root_dir/BAFLineDP")

get.file.level.metrics <- function(df) {
  all.gt <- df$file.level.ground.truth
  all.prob <- df$prediction.prob
  all.pred <- df$prediction.label

  all.pred <- as.factor(all.pred)
  all.gt <- as.factor(all.gt)

  confusion.mat <- confusionMatrix(all.pred, reference = all.gt)

  BA <- confusion.mat$byClass["Balanced Accuracy"]
  AUC <- pROC::auc(all.gt, all.prob)

  all.pred <- ifelse(all.pred == "False", 0, ifelse(all.pred == "True", 1, all.pred))
  all.gt <- ifelse(all.gt == "False", 0, ifelse(all.gt == "True", 1, all.gt))

  MCC <- mcc(all.gt, all.pred, cutoff = 0.5)

  if (is.nan(MCC)) {
    MCC <- 0
  }

  file.eval.result <- c(AUC, MCC, BA)

  return(file.eval.result)
}

get.line.level.metrics <- function(df) {
  sorted <- df %>%
    filter(file.level.ground.truth == "True" & prediction.label == "True") %>%
    group_by(test, filename) %>%
    arrange(-line.attention.score, .by_group = TRUE) %>%
    mutate(order = row_number()) %>%
    ungroup()

  total_true <- sorted %>%
    group_by(test, filename) %>%
    summarize(total_true = sum(line.level.ground.truth == "True")) %>%
    ungroup()

  # calculate Recall@Top20%LOC
  Recall20LOC <- sorted %>%
    group_by(test, filename) %>%
    mutate(effort = round(order / n(), digits = 2 )) %>%
    filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>%
    mutate(Recall20LOC = correct_pred / total_true) %>%
    ungroup()

  # calculate Effort@Top20%Recall
  Effort20Recall <- sorted %>%
    merge(total_true) %>%
    group_by(test, filename) %>%
    mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cummulative_correct_pred / total_true, digits = 2)) %>%
    summarise(Effort20Recall = sum(recall <= 0.2) / n()) %>%
    ungroup()

  Recall20LOC$Target <- sub('-.*', '', Recall20LOC$test)
  Effort20Recall$Target <- sub('-.*', '', Effort20Recall$test)

  line.eval.result <- list(Recall20LOC, Effort20Recall)

  return(line.eval.result)
}

get.file.results <- function (prediction.dir, mode, src = NULL) {
  all.files <- list.files(prediction.dir)
  all.auc <- NULL
  all.mcc <- NULL
  all.ba <- NULL
  all.target <- NULL

  pb <- progress_bar$new(
    format = 'File level metric calculation [:bar] :percent eta: :eta',
    total = length(all.files),
    clear = FALSE,
    width = 80
  )

  for (f in all.files) {
    df <- read.csv(paste0(prediction.dir, f))
    df <- as_tibble(df)
    df <- select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
    df <- distinct(df)

    file.level.result <- get.file.level.metrics(df)
    all.auc <- append(all.auc, file.level.result[1])
    all.mcc <- append(all.mcc, file.level.result[2])
    all.ba <- append(all.ba, file.level.result[3])

    f.split <- strsplit(f, '-')[[1]]

    if (mode == 'within') {
      all.target <- append(all.target, f.split[1])
    } else {
      all.target <- append(all.target, f.split[length(f.split) - 1])
    }

    pb$tick()
    Sys.sleep(0.05)
  }

  results <- data.frame(Target = all.target, AUC = all.auc, MCC = all.mcc, BA = all.ba)

  if (!is.null(src)) {
    results <- data.frame(Src = src, results)
  } else {
    results <- results %>%
      group_by(Target) %>%
      summarise(AUC = mean(AUC), MCC = mean(MCC), BA = mean(BA)) %>%
      ungroup()
  }

  return(results)
}

get.line.results <- function (prediction.dir, mode, src = NULL) {
  all.files <- list.files(prediction.dir)
  df.all <- NULL

  pb <- progress_bar$new(
    format = 'Line level datas loading [:bar] :percent eta: :eta',
    total = length(all.files),
    clear = FALSE,
    width = 80
  )

  for (f in all.files) {
    df <- read.csv(paste0(prediction.dir, f))
    df.all <- rbind(df.all, df)

    pb$tick()
    Sys.sleep(0.05)
  }

  # set the line attention of comment lines to 0
  df.all[df.all$is.comment.line == "True",]$line.attention.score <- 0

  line.level.result <- get.line.level.metrics(df.all)
  Recall20LOC <- line.level.result[[2]]
  Effort20Recall <- line.level.result[[3]]

  if (mode == 'cross' && !is.null(src)) {
    results <- data.frame(
      Src = src,
      Target = Recall20LOC$Target,
      Recall20LOC = Recall20LOC$Recall20LOC,
      Effort20Recall = Effort20Recall$Effort20Recall
    )
  } else {
    Recall20LOC <- Recall20LOC %>%
      group_by(Target) %>%
      summarise(Recall20LOC = mean(Recall20LOC)) %>%
      ungroup()

    Effort20Recall <- Effort20Recall %>%
      group_by(Target) %>%
      summarise(Effort20Recall = mean(Effort20Recall)) %>%
      ungroup()

    results <- data.frame(
      Target = Recall20LOC$Target,
      Recall20LOC = Recall20LOC$Recall20LOC,
      Effort20Recall = Effort20Recall$Effort20Recall
    )
  }

  return(results)
}

# compute the results of WPDP
prediction.dir.within <- 'output/prediction/BAFLineDP/within-release/'
print('In-domain result processing...')
within.file.level.results <- get.file.results(prediction.dir.within, 'within')
within.line.level.results <- get.line.results(prediction.dir.within, 'within')

# compute the results of CPDP
prediction.dir.cross <- 'output/prediction/BAFLineDP/cross-release/'
print('Cross-domain result processing...')
cross.file.level.results <- NULL
cross.line.level.results <- NULL

projs <- list.files(prediction.dir.cross)
for (p in projs) {
  actual.prediction.dir <- paste0(prediction.dir.cross, p, '/')
  cross.file.level.results <- rbind(cross.file.level.results, get.file.results(actual.prediction.dir, 'cross', p))
  cross.line.level.results <- rbind(cross.line.level.results, get.line.results(actual.prediction.dir, 'cross', p))
  print(paste0('Finished ', p))
}

cross.file.level.results <- cross.file.level.results %>%
  group_by(Target) %>%
  summarise(AUC = mean(AUC), MCC = mean(MCC), BA = mean(BA)) %>%
  ungroup()

cross.line.level.results <- cross.line.level.results %>%
  group_by(Target) %>%
  summarise(Recall20LOC = mean(Recall20LOC), Effort20Recall = mean(Effort20Recall)) %>%
  ungroup()

save_path <- 'output/result/BAFLineDP/'
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

within.results <- merge(within.file.level.results, within.line.level.results)
cross.results <- merge(cross.file.level.results, cross.line.level.results)

write.csv(within.results, file = paste0(save_path, "within_results.csv"), row.names = FALSE)
write.csv(cross.results, file = paste0(save_path, "cross_results.csv"), row.names = FALSE)