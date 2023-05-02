import json
from tqdm import tqdm
if __name__=="__main__":
    douban_path = "/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/data/douban/train/data.json"
    ecommerce_path = "/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/data/ecommerce/train/data.json"
    rrs_path = "/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/data/rrs/train/data.json"
    save_path = "/mmu_nlp/wuxing/suzhenpeng/cotmae-dial/data/chinese_mix/train/data.json"
    f = open(douban_path,"r")    
    douban = [ json.loads(i) for i in f ]
    # f.close()

    f = open(ecommerce_path,"r")    
    ecommerce = [ json.loads(i) for i in f ]
    # f.close()

    f = open(rrs_path,"r")    
    rrs = [ json.loads(i) for i in f ]
    # f.close()

    with open(save_path, 'w') as f:
        douban_num,ecommerce_num,rrs_num=0,0,0
        for i in tqdm(douban):
            douban_num += 1
            f.write(json.dumps(i) + '\n')

        for i in tqdm(ecommerce):
            ecommerce_num += 1
            f.write(json.dumps(i) + '\n')

        for i in tqdm(rrs):
            rrs_num += 1
            f.write(json.dumps(i) + '\n')

    print(douban_num,ecommerce_num,rrs_num)
        


        