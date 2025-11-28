# 实体消歧 - 从头开始处理测试目录中的所有文件
python main_cli.py --task entity_disambiguation \
    --disambiguation_input_dir "E:\KKGG\output\KG\test" \
    --disambiguation_output_dir "E:\KKGG\output\KG\test_削岐后" \
    --progress_file "E:\KKGG\project\process.json" \
    --start_index 0 \
    --end_index -1 \
    --log_level INFO

# 概念聚类 - 从头开始处理测试目录中的所有文件
python main_cli.py --task concept_clustering \
    --clustering_input_dir "E:\KKGG\output\KG\test" \
    --cluster_output_file "E:\KKGG\output\terms\test_entity_cluster_triples.json" \
    --progress_file "E:\KKGG\project\process.json" \
    --start_index 0 \
    --end_index -1 \
    --log_level INFO