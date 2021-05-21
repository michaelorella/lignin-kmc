package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 SYR
    residue 2 SYR
    residue 3 GUAI
    residue 4 SYR
    residue 5 SYR
    residue 6 SYR
    residue 7 GUAI
    residue 8 SYR
}
patch BO4 L:3 L:7
patch BB L:2 L:8
patch BB L:1 L:4
patch BB L:5 L:6
patch 4O5 L:2 L:3
regenerate angles dihedrals
writepsf lignin.psf
