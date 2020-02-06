package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 S
    residue 2 S
    residue 3 S
    residue 4 S
    residue 5 G
    residue 6 S
    residue 7 S
    residue 8 S
    residue 9 S
    residue 10 S
    residue 11 S
    residue 12 G
    residue 13 S
    residue 14 S
    residue 15 S
    residue 16 S
    residue 17 S
    residue 18 S
    residue 19 S
    residue 20 S
    residue 21 S
    residue 22 S
    residue 23 S
    residue 24 S
}
patch BB L:10 L:24
patch BB L:2 L:14
patch BB L:8 L:18
patch BB L:20 L:23
patch B5G L:15 L:5
patch BB L:1 L:9
patch BB L:4 L:19
patch BB L:7 L:13
patch BB L:17 L:21
patch BB L:11 L:16
patch BO4 L:12 L:24
patch BB L:3 L:22
patch BO4 L:6 L:17
regenerate angles dihedrals
writepsf lignin.psf
