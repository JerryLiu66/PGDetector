## PGDetector

This repository includes the code and the data of paper “Fishing for Fraudsters: Uncovering Ethereum Phishing Gangs with Blockchain Data”.

* Before you use the code, you should prepare the seeds to search like 'data/APPR_scamdb_2056.xlsx', use BlockchainSpider https://github.com/wuzhy1ng/BlockchainSpider to search the related accounts of each seed, and prepare the lifetime of accounts in 'data/lifetime'.
* Run motif_statistics.py to calculate the motif information based on the search subgraph.
* Run PGDetector_main.py to search the local community around each seed.
* Run merge_with_clustering.py to merge the communities when given multiple seeds.

You can also customize the parameters at params.json.

