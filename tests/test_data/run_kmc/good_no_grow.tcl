package require psfgen
topology toppar/top_all36_cgenff.rtf
topology toppar/top_lignin.top
segment L {
    residue 1 SYR
    residue 2 SYR
    residue 3 SYR
    residue 4 SYR
    residue 5 GUAI
    residue 6 SYR
    residue 7 SYR
    residue 8 SYR
    residue 9 SYR
    residue 10 SYR
    residue 11 SYR
    residue 12 GUAI
    residue 13 SYR
    residue 14 SYR
    residue 15 SYR
    residue 16 SYR
    residue 17 SYR
    residue 18 SYR
    residue 19 SYR
    residue 20 SYR
    residue 21 SYR
    residue 22 SYR
    residue 23 SYR
    residue 24 SYR
}
patch BB L:8 L:10
patch BB L:2 L:15
patch BB L:3 L:11
patch BB L:4 L:21
patch BB L:1 L:24
patch BB L:9 L:16
patch BB L:13 L:17
patch BB L:18 L:20
patch BO4 L:5 L:15
patch BB L:19 L:23
patch BB L:6 L:7
patch BB L:14 L:22
patch BO4 L:12 L:17
regenerate angles dihedrals
writepsf lignin.psf
