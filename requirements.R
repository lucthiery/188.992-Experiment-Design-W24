# Erstelle ein lokales Library-Verzeichnis (falls nicht vorhanden)
library_path <- file.path(Sys.getenv("HOME"), "R_packages")
dir.create(library_path, showWarnings = FALSE, recursive = TRUE)

# Setze das Library-Verzeichnis
.libPaths(library_path)

# Liste der benötigten Pakete
packages <- c("text2vec", "e1071", "caTools", "tidyr")

install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, dependencies = TRUE, lib = library_path)
    }
  }
}

install_if_missing(packages)

# Library-Pfad ausgeben (zum Debugging)
print(paste("R packages installiert in:", library_path))
