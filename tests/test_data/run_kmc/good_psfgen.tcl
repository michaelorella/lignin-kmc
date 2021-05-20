package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 SYR
    residue 2 SYR
    residue 3 GUAI
    residue 4 GUAI
    residue 5 GUAI
    residue 6 GUAI
    residue 7 SYR
    residue 8 SYR
    residue 9 GUAI
    residue 10 SYR
}
patch BB L:1 L:2
patch BO4 L:3 L:2
patch BO4 L:4 L:3
patch B5G L:5 L:4
patch B5G L:6 L:5
patch BO4 L:7 L:6
patch BB L:8 L:9
patch 4O5 L:7 L:9
patch BO4 L:10 L:8
regenerate angles dihedrals
writepsf lignin.psf
