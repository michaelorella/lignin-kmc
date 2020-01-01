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
    residue 9 S
    residue 10 S
    residue 11 G
    residue 12 S
    residue 13 S
    residue 14 G
    residue 15 S
    residue 16 G
    residue 17 S
    residue 18 G
    residue 19 G
    residue 20 S
    residue 21 S
    residue 22 S
    residue 23 S
    residue 24 S
    residue 25 S
    residue 26 S
    residue 27 S
    residue 28 S
    residue 29 S
    residue 30 S
    residue 31 G
    residue 32 S
    residue 33 G
    residue 34 S
    residue 35 S
    residue 36 S
    residue 37 S
    residue 38 S
    residue 39 S
    residue 40 S
    residue 41 S
    residue 42 S
    residue 43 S
    residue 44 S
    residue 45 S
    residue 46 S
    residue 47 S
    residue 48 G
    residue 49 S
    residue 50 S
    residue 51 S
    residue 52 S
    residue 53 G
    residue 54 S
    residue 55 S
    residue 56 S
    residue 57 S
    residue 58 S
    residue 59 S
    residue 60 S
    residue 61 S
    residue 62 S
    residue 63 G
    residue 64 S
    residue 65 S
    residue 66 S
    residue 67 S
}
patch BB L:21 L:47
patch B5G L:48 L:3
patch BB L:39 L:51
patch B5G L:19 L:33
patch B5G L:63 L:16
patch B5G L:67 L:31
patch BB L:1 L:41
patch B5G L:56 L:7
patch BB L:12 L:23
patch BB L:36 L:44
patch BB L:26 L:28
patch BO4 L:53 L:23
patch BB L:43 L:52
patch BB L:58 L:61
patch BB L:22 L:59
patch BB L:49 L:60
patch BB L:42 L:66
patch BB L:25 L:32
patch BB L:35 L:57
patch BB L:38 L:46
patch BB L:10 L:55
patch BB L:9 L:37
patch BO4 L:18 L:9
patch BB L:27 L:54
patch BB L:8 L:65
patch BB L:24 L:62
patch BB L:13 L:17
patch 4O4 L:59 L:19
patch 4O4 L:65 L:48
patch BB L:50 L:64
patch B5G L:5 L:11
patch BB L:20 L:30
patch BB L:29 L:45
patch BO4 L:14 L:47
patch BB L:34 L:40
patch 4O4 L:67 L:63
patch O4AL L:63
patch BO4 L:2 L:48
patch BO4 L:15 L:19
patch B1 L:4 L:6
patch 4O4 L:13 L:14
patch 4O4 L:42 L:18
patch 4O4 L:38 L:53
regenerate angles dihedrals
writepsf lignin.psf
