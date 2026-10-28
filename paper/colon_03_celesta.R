# CELESTA on every region of the published colon table with its own bundled prior.
# Per region (CELESTA is designed per image): GMM marker-activation probabilities, anchor + iterative assignment.
# Saves per cell: Final cell type and the 18 marker activation probabilities (used to binarise for QuantCell).
suppressMessages({library(zeallot); library(Rmixmod); library(spdep); library(CELESTA)})
C <- file.path(Sys.getenv("QC_OUT"), "celesta"); dir.create(file.path(C, "colon/per_region_v2"), showWarnings=FALSE, recursive=TRUE); setwd(file.path(C, "colon/per_region_v2"))
prior <- read.csv(file.path(Sys.getenv("QC_INPUTS"), "celesta_colon_prior_marker_info.csv"))   # CELESTA published colon signature
df <- read.csv(file.path(Sys.getenv("QC_EXTERNAL"), "colon/CRC_clusters_neighborhoods_markers.csv"), check.names=TRUE)
names(df)[names(df)=="X.X"] <- "X"; names(df)[names(df)=="Y.Y"] <- "Y"
markers <- colnames(prior)[3:ncol(prior)]; stopifnot(all(markers %in% colnames(df)))
regs <- sort(unique(df$Region)); cat(sprintf("%d cells, %d regions, %d markers matched\n", nrow(df), length(regs), length(markers)))
t0 <- Sys.time(); done <- 0
for (r in regs) {
  out <- sprintf("region_%s.csv", r); if (file.exists(out)) next
  sub <- df[df$Region == r, c("CellID","Region","X","Y", markers)]
  res <- tryCatch({
    obj <- CreateCelestaObject(project_title=paste0("reg", r), prior_marker_info=prior, imaging_data_file=sub)
    obj <- FilterCells(obj, high_marker_threshold=0.9, low_marker_threshold=0.4)
    obj <- AssignCells(obj, max_iteration=10, cell_change_threshold=0.01, save_result=FALSE)
    fa <- obj@final_cell_type_assignment; prob <- as.data.frame(obj@marker_exp_prob); colnames(prob) <- paste0("prob_", markers)
    an <- obj@anchor_cell_type_assignment; if (is.null(dim(an))) an <- matrix(an, ncol=1)
    anchor_num <- apply(an, 1, function(v) { nz <- v[v != 0]; if (length(nz)) tail(nz, 1) else 0 })   # deepest anchor level, 0 = not an anchor
    anchor <- ifelse(anchor_num == 0, "none", prior[anchor_num, 1])
    cbind(sub[, c("CellID","Region")], final_cell_type=fa[, "Final cell type"], anchor=anchor, prob)
  }, error=function(e) { cat(sprintf("region %s FAILED: %s\n", r, conditionMessage(e))); NULL })
  if (!is.null(res)) { write.csv(res, out, row.names=FALSE); done <- done + 1 }
  if (done %% 20 == 0) cat(sprintf("  %d regions done, %.0f s\n", done, as.numeric(difftime(Sys.time(), t0, units="secs"))))
}
files <- list.files(pattern="^region_.*csv$"); all <- do.call(rbind, lapply(files, read.csv))
write.csv(all, paste0(C, "/colon/celesta_all_regions_v2.csv"), row.names=FALSE)
cat(sprintf("DONE: %d regions, %d cells, %.0f s\n", length(files), nrow(all), as.numeric(difftime(Sys.time(), t0, units="secs"))))
print(table(all$final_cell_type))
