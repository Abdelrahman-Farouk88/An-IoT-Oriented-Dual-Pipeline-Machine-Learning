# NASA C-MAPSS turbofan data

Place the original FD001-FD004 **training** files here:

```text
train_FD001.txt
train_FD002.txt
train_FD003.txt
train_FD004.txt
```

The scripts use the whitespace-separated files distributed by the NASA Prognostics Center of Excellence (PCoE) data repository. Each row contains the unit number, cycle, three operational settings, and 21 sensor measurements; the code assigns the standard column names.

The manuscript uses `train_FD001.txt` for model development and the FD002-FD004 training files for external domain-shift tests. Test/RUL files are not required by the scripts in this release.

C-MAPSS is a simulation-generated benchmark. It is not field data, and no claim of field validation is made.

The data are not redistributed in this release. Obtain them from the NASA PCoE repository and follow its terms of use.
