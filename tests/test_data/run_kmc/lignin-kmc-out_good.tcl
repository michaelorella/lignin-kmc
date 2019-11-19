package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 G
    residue 2 S
    residue 3 S
    residue 4 G
    residue 5 S
    residue 6 S
    residue 7 S
    residue 8 S
    residue 9 G
    residue 10 S
}
patch B5G L:2 L:1
patch BO4 L:3 L:2
patch B5G L:5 L:4
patch BB L:6 L:7
patch BO4 L:8 L:5
patch BO4 L:9 L:7
patch 4O4 L:3 L:9
patch BO4 L:10 L:9
regenerate angles dihedrals
writepsf L.psf
