# CELESTA on the bone-marrow codex, per section, with the paper's own marker combinations as the prior.
suppressMessages({library(zeallot); library(Rmixmod); library(spdep); library(CELESTA)})
B <- file.path(Sys.getenv("QC_OUT"), "retrain/benchmarks/CELESTA"); dir.create(file.path(B, "per_section"), showWarnings=FALSE, recursive=TRUE); setwd(file.path(B, "per_section"))
prior <- read.csv(paste0(B, "/prior_marker_info.csv"), check.names=TRUE)
df <- read.csv(paste0(B, "/imaging_data.csv"), check.names=TRUE)
markers <- colnames(prior)[3:ncol(prior)]; stopifnot(all(markers %in% colnames(df)))
secs <- sort(unique(df$section)); cat(sprintf("%d cells, %d sections, %d markers matched, %d types\n", nrow(df), length(secs), length(markers), nrow(prior)))
t0 <- Sys.time()
for (s in secs) {
  out <- sprintf("section_%s.csv", s); if (file.exists(out)) next
  sub <- df[df$section == s, c("CellID","section","X","Y", markers)]
  res <- tryCatch({
    obj <- CreateCelestaObject(project_title=s, prior_marker_info=prior, imaging_data_file=sub)
    obj <- FilterCells(obj, high_marker_threshold=0.9, low_marker_threshold=0.4)
    obj <- AssignCells(obj, max_iteration=10, cell_change_threshold=0.01, save_result=FALSE)
    fa <- obj@final_cell_type_assignment; prob <- as.data.frame(obj@marker_exp_prob); colnames(prob) <- paste0("prob_", markers)
    cbind(sub[, c("CellID","section")], final_cell_type=fa[, "Final cell type"], prob)
  }, error=function(e) { cat(sprintf("section %s FAILED: %s\n", s, conditionMessage(e))); NULL })
  if (!is.null(res)) write.csv(res, out, row.names=FALSE)
  cat(sprintf("  %s done (%d cells), %.0f s\n", s, nrow(sub), as.numeric(difftime(Sys.time(), t0, units="secs"))))
}
files <- list.files(pattern="^section_.*csv$"); all <- do.call(rbind, lapply(files, read.csv))
write.csv(all, paste0(B, "/celesta_bm_all.csv"), row.names=FALSE)
cat(sprintf("DONE: %d sections, %d cells, %.0f s\n", length(files), nrow(all), as.numeric(difftime(Sys.time(), t0, units="secs"))))
print(table(all$final_cell_type))
