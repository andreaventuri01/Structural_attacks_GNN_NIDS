#! /bin/bash

ctu_dir="datasets/CTU/"
ton_dir="datasets/ToN-IoT/"

preproc_dir="preprocessed_data/"

for attack in neris rbot virut menti murlo
do
    echo -------------------------------------
    echo ---------BOTNET: ${attack}-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    python preprocessing.py --mal ${ctu_dir}${attack}.csv --ben ${ctu_dir}benign_tot.csv 
    
    echo -------Training E-GRAPHSAGE----------
    python train_egraphsage.py --train ${preproc_dir}CTU/${attack}_train.csv --test ${preproc_dir}CTU/${attack}_test.csv
    
    echo -------Testing E-GRAPHSAGE-----------
    python baseline_test_egraphsage.py --test ${preproc_dir}CTU/${attack}_test.csv
    python baseline_test_egraphsage.py --test ${preproc_dir}CTU/${attack}_test.csv --feature_attack
    python baseline_test_egraphsage.py --test ${preproc_dir}CTU/${attack}_test.csv --structure_attack benign_from_C
    python baseline_test_egraphsage.py --test ${preproc_dir}CTU/${attack}_test.csv --structure_attack malicious_from_C
    python baseline_test_egraphsage.py --test ${preproc_dir}CTU/${attack}_test.csv --structure_attack add_node

    echo -------Training LINE-GRAPHSAGE-------
    python batch_train_linegraphsage.py --train ${preproc_dir}CTU/${attack}_train.csv --test ${preproc_dir}CTU/${attack}_test.csv
    
     echo -------Testing LINE-GRAPHSAGE--------
    python baseline_test_linegraph.py --test ${preproc_dir}CTU/${attack}_test.csv
    python baseline_test_linegraph.py --test ${preproc_dir}CTU/${attack}_test.csv --feature_attack
    python baseline_test_linegraph.py --test ${preproc_dir}CTU/${attack}_test.csv --structure_attack benign_from_C
    python baseline_test_linegraph.py --test ${preproc_dir}CTU/${attack}_test.csv --structure_attack malicious_from_C
    python baseline_test_linegraph.py --test ${preproc_dir}CTU/${attack}_test.csv --structure_attack add_node
done

for attack in backdoor ddos dos injection password ransomware scanning xss
do
    echo -------------------------------------
    echo ---------ATTACK: ${attack}-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    python preprocessing_ToN.py --mal ${ton_dir}${attack}.csv --ben ${ton_dir}benign_tot.csv 
    
    echo -------Training E-GRAPHSAGE----------
    python train_egraphsage.py --train ${preproc_dir}ToN_IoT/${attack}_train.csv --test ${preproc_dir}ToN_IoT/${attack}_test.csv
    
    echo -------Testing E-GRAPHSAGE-----------
    python baseline_test_egraphsage.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv
    python baseline_test_egraphsage.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --feature_attack
    python baseline_test_egraphsage.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --structure_attack benign_from_C
    python baseline_test_egraphsage.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --structure_attack malicious_from_C
    python baseline_test_egraphsage.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --structure_attack add_node

    echo -------Training LINE-GRAPHSAGE-------
    python batch_train_linegraphsage.py --train ${preproc_dir}ToN_IoT/${attack}_train.csv --test ${preproc_dir}ToN_IoT/${attack}_test.csv
    
    echo -------Testing LINE-GRAPHSAGE--------
    python baseline_test_linegraph.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv
    python baseline_test_linegraph.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --feature_attack
    python baseline_test_linegraph.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --structure_attack benign_from_C
    python baseline_test_linegraph.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --structure_attack malicious_from_C
    python baseline_test_linegraph.py --test ${preproc_dir}ToN_IoT/${attack}_test.csv --structure_attack add_node
done
