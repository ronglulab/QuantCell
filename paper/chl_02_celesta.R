# CELESTA on a cHL dataset (cHL1 or cHL CODEX) from the MAPS positive-only marker list; CELESTA_B = dataset folder. Patch: CELESTA:::GetScore slices the activation matrix with
# `[, non_NA_index]`, which R drops to a vector when a cell type has a single marker (Endothelial, Neutrophil here)
# and apply() then fails with "dim(X) must have a positive length". Re-slice with drop=FALSE; logic otherwise identical.
suppressMessages({library(zeallot); library(Rmixmod); library(spdep); library(CELESTA)})
gs <- CELESTA:::GetScore; body_txt <- deparse(body(gs)); body_txt <- gsub("activation_prob_to_use[, non_NA_index]", "activation_prob_to_use[, non_NA_index, drop=FALSE]", body_txt, fixed=TRUE)
body(gs) <- parse(text=paste(body_txt, collapse="\n"))[[1]]; environment(gs) <- asNamespace("CELESTA"); assignInNamespace("GetScore", gs, ns="CELESTA")
PRIOR <- Sys.getenv("CELESTA_PRIOR", "prior_marker_info.csv"); SUB <- Sys.getenv("CELESTA_OUT", "per_fov")
B <- Sys.getenv("CELESTA_B"); dir.create(paste0(B, "/", SUB), showWarnings=FALSE); setwd(paste0(B, "/", SUB))
prior <- read.csv(paste0(B, "/", PRIOR), check.names=TRUE); df <- read.csv(paste0(B, "/imaging_data.csv"), check.names=TRUE)
markers <- colnames(prior)[3:ncol(prior)]; stopifnot(all(markers %in% colnames(df))); t0 <- Sys.time(); cat(sprintf("prior %s -> %s\n", PRIOR, SUB))
for (f in sort(unique(df$sample))) {
  out <- sprintf("s_%s.csv", gsub("[^A-Za-z0-9]","_",f)); if (file.exists(out)) next
  sub <- df[df$sample == f, c("CellID","sample","X","Y", markers)]
  res <- tryCatch({ obj <- CreateCelestaObject(project_title=paste0("fov",f), prior_marker_info=prior, imaging_data_file=sub)
    obj <- FilterCells(obj, high_marker_threshold=0.9, low_marker_threshold=0.4); obj <- AssignCells(obj, max_iteration=10, cell_change_threshold=0.01, save_result=FALSE)
    cbind(sub[, c("CellID","sample")], final_cell_type=obj@final_cell_type_assignment[, "Final cell type"]) }, error=function(e) { cat(sprintf("fov %s FAILED: %s\n", f, conditionMessage(e))); NULL })
  if (!is.null(res)) write.csv(res, out, row.names=FALSE); cat(sprintf("  fov %s %s (%d cells), %.0f s\n", f, if (is.null(res)) "FAILED" else "done", nrow(sub), as.numeric(difftime(Sys.time(), t0, units="secs"))))
}
all <- do.call(rbind, lapply(list.files(pattern="^s_.*csv$"), read.csv)); write.csv(all, paste0(B, "/celesta_all.csv"), row.names=FALSE)
cat(sprintf("DONE: %d cells, %.0f s\n", nrow(all), as.numeric(difftime(Sys.time(), t0, units="secs")))); print(table(all$final_cell_type))
