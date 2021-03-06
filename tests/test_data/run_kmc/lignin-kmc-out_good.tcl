package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 G
    residue 2 G
    residue 3 G
    residue 4 S
    residue 5 S
    residue 6 S
    residue 7 G
    residue 8 S
    residue 9 G
    residue 10 S
}
patch B5G L:1 L:2
patch BO4 L:3 L:1
patch B5G L:4 L:3
patch BO4 L:5 L:4
patch BO4 L:6 L:5
patch BO4 L:7 L:6
patch BO4 L:8 L:7
patch BO4 L:9 L:8
patch B5G L:10 L:9
regenerate angles dihedrals
writepsf lignin.psf
