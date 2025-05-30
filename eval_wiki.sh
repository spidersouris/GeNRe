#!/bin/bash
echo "Checking if the required directories exist..."
if [ ! -d "_allscores_sbert/wiki" ]; then
    echo "Creating directories..."
    mkdir -p _allscores_sbert/wiki
fi

echo "Running evaluation for Wiki dataset..."
echo "Evaluating column 2..."
python eval.py -c "gen" -ac 2 data/eval/wiki/gen/wiki_gen_eval.csv _allscores_sbert/wiki/wiki_gen_eval_out_2.csv
echo "Evaluating column 3..."
python eval.py -c "gen" -ac 3 data/eval/wiki/gen/wiki_gen_eval.csv _allscores_sbert/wiki/wiki_gen_eval_out_3.csv
echo "Evaluating column 4..."
python eval.py -c "gen" -ac 4 data/eval/wiki/gen/wiki_gen_eval.csv _allscores_sbert/wiki/wiki_gen_eval_out_4.csv
echo "Evaluating column 11..."
python eval.py -c "gen" -ac 11 data/eval/wiki/gen/wiki_gen_eval.csv _allscores_sbert/wiki/wiki_gen_eval_out_11.csv
echo "Evaluating column 12..."
python eval.py -c "gen" -ac 12 data/eval/wiki/gen/wiki_gen_eval.csv _allscores_sbert/wiki/wiki_gen_eval_out_12.csv
echo "Evaluating column 13..."
python eval.py -c "gen" -ac 13 data/eval/wiki/gen/wiki_gen_eval.csv _allscores_sbert/wiki/wiki_gen_eval_out_13.csv