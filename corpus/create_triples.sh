#!/usr/bin/env bash

wiki_dump_file=$1

# Parse wikipedia. Splitting to 20 files and running in parallel.
echo 'Parsing wikipedia...'
split -nl/20 $wiki_dump_file $wiki_dump_file"_";

for x in {a..t}
do
( python parse_wikipedia.py $wiki_dump_file"_a"$x $wiki_dump_file"_a"$x"_parsed" ) &
done
wait

python term_to_id_make.py $wiki_dump_file

for x in {a..t}
do
( awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_a"$x"_parsed" > paths"_a"$x ) &
done
wait

cat paths_a* > paths_temp;
cat paths_temp | grep -v "$(printf '\t1$')" > frequent_paths_temp;
awk -F$'\t' '{i[$1]+=$2} END{for(x in i){print x"\t"i[x]}}' frequent_paths_temp > paths;
awk -F$'\t' '$2 >= 5 {print $1}' paths > frequent_paths

python frequent_path2id.py

python pathfile2id.py $wiki_dump_file

cat $wiki_dump_file"_a"*"_parsed_id" > id_triples