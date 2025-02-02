# Erstelle ein lokales Library-Verzeichnis (falls nicht vorhanden)
library_path <- file.path(Sys.getenv("HOME"), "R_packages")
dir.create(library_path, showWarnings = FALSE, recursive = TRUE)

# Setze das Library-Verzeichnis
.libPaths(library_path)

# Liste der benötigten Pakete (Matrix mit aufnehmen!)
packages <- c("Matrix", "text2vec", "e1071", "caTools", "tidyr")

# Funktion, die fehlende Pakete installiert
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg,
                       dependencies = TRUE,
                       lib = library_path,
                       repos = "https://cloud.r-project.org")
    }
  }
}

# Installation fehlender Pakete
install_if_missing(packages)

# Print Library-Pfad for debugging
print(paste("R packages installed in:", library_path))
