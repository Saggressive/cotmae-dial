export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
#douban
# python make_data.py \
#     --tokenizer bert-base-chinese \
#     --save_to data/douban \
#     --data_path data/douban

#ecommerce
python make_data.py \
    --tokenizer bert-base-chinese \
    --save_to data/ecommerce \
    --data_path data/ecommerce

#RRS
# python make_data.py \
#     --tokenizer bert-base-chinese \
#     --save_to data/rrs \
#     --data_path data/rrs