package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 S
    residue 2 S
    residue 3 G
    residue 4 S
    residue 5 S
    residue 6 S
    residue 7 G
    residue 8 S
}
patch B5G L:1 L:3
patch BB L:2 L:6
patch BB L:4 L:8
patch BO4 L:7 L:2
patch BO4 L:5 L:8
patch 4O4 L:5 L:7
regenerate angles dihedrals
writepsf lignin.psf
