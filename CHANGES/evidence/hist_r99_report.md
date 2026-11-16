# r99 history A/B — off / on-as-shipped / on+new detection+additive

## dmr (DMR (the 130 pre-registered ids)) n=130, rows=390, spend $0.5765
- off        acc 0.7923 (103/130) mean_score 0.6544
- on_shipped acc 0.6692 (87/130) mean_score 0.5945
- new        acc 0.7692 (100/130) mean_score 0.6379
- off -> on_shipped: b=3 c=19 n=130 p=0.0009
- off -> new: b=1 c=4 n=130 p=0.3750
- on_shipped -> new: b=19 c=6 n=130 p=0.0146

## beam (BEAM event_ordering + temporal_reasoning) n=80, rows=240, spend $0.5207
- off        acc 0.55 (44/80) mean_score 0.3846
- on_shipped acc 0.35 (28/80) mean_score 0.2828
- new        acc 0.5875 (47/80) mean_score 0.4302
- off -> on_shipped: b=3 c=19 n=80 p=0.0009
- off -> new: b=7 c=4 n=80 p=0.5488
- on_shipped -> new: b=20 c=1 n=80 p=0.0000

## locomo (LoCoMo temporal) n=321, rows=963, spend $0.3035
- off        acc 0.324 (104/321) mean_score 0.138
- on_shipped acc 0.4704 (151/321) mean_score 0.1229
- new        acc 0.4393 (141/321) mean_score 0.1408
- off -> on_shipped: b=63 c=16 n=321 p=0.0000
- off -> new: b=46 c=9 n=321 p=0.0000
- on_shipped -> new: b=17 c=27 n=321 p=0.1742

## fired (time-question detection)
- dmr off        21/130
- dmr on_shipped 130/130
- dmr new        21/130
- beam off        79/80
- beam on_shipped 73/80
- beam new        79/80
- locomo off        320/321
- locomo on_shipped 294/321
- locomo new        320/321
