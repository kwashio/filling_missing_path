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

#
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_aa_parsed" > "paths_aa"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ab_parsed" > "paths_ab"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ac_parsed" > "paths_ac"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ad_parsed" > "paths_ad"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ae_parsed" > "paths_ae"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_af_parsed" > "paths_af"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ag_parsed" > "paths_ag"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ah_parsed" > "paths_ah"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ai_parsed" > "paths_ai"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_aj_parsed" > "paths_aj"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ak_parsed" > "paths_ak"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_al_parsed" > "paths_al"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_am_parsed" > "paths_am"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_an_parsed" > "paths_an"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ao_parsed" > "paths_ao"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$wiki_dump_file""_ap_parsed" > "paths_ap"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ab_parsed.out" > "paths_ab"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ac_parsed.out" > "paths_ac"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ad_parsed.out" > "paths_ad"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ae_parsed.out" > "paths_ae"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_af_parsed.out" > "paths_af"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ag_parsed.out" > "paths_ag"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ah_parsed.out" > "paths_ah"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ai_parsed.out" > "paths_ai"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_aj_parsed.out" > "paths_aj"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ak_parsed.out" > "paths_ak"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_al_parsed.out" > "paths_al"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_am_parsed.out" > "paths_am"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_an_parsed.out" > "paths_an"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ao_parsed.out" > "paths_ao"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ap_parsed.out" > "paths_ap"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_aq_parsed.out" > "paths_aq"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_ar_parsed.out" > "paths_ar"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_as_parsed.out" > "paths_as"
#awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "wiki_at_parsed.out" > "paths_at"
cat paths_a* > paths_temp;
cat paths_temp | grep -v "$(printf '\t1$')" > frequent_paths_temp;
awk -F$'\t' '{i[$1]+=$2} END{for(x in i){print x"\t"i[x]}}' frequent_paths_temp > paths;
awk -F$'\t' '$2 >= 5 {print $1}' paths > frequent_paths