import json
import copy


config = {}
base = {"tags":{"base_model":"random_forest"},
        "params":{},
        "variables":[],
        "productos":[]}


clusters = {"0": ['11', '14', '17', '18', '19', '24', '5', '9'],
            "1": ['12', '21', '22', '26', '30', '31', '32', '35', 
            '36', '38', '40', '42', '43', '45', '46', '48', '49', 
            '50', '52', '53', '55', '59', '60', '61', '62', '63',
             '64', '66', '67', '68', '69', '72', '77', '79', '84',
              '87', '94', '96', '98'],
            "2": ['37', '57', '58', '73', '78', '85', '88', '89', '91', '97']}

all_products = ['1','10','11','12','13','14','15','17','18','19',
                '2','20','21','22','23','24','26','28','29',
                '3','30','31','32','35','36','37','38',
                '4','40','42','43','45','46','48','49',
                '5','50','52','53','55','57','58','59',
                '60','61','62','63','64','66','67','68','69',
                '72','73','74','77','78','79',
                '80','84','85','87','88','89',
                '9','91','94','95','96','97','98','99']


config["ALL"] = base
config["ALL"]["tags"]["type"] = "all"
config["ALL"]["productos"] = all_products


for cluster in list(clusters):
    key = "CL_" + cluster
    new = copy.deepcopy(base)
    new["productos"] = clusters[cluster]
    new["tags"]["type"] = "cluster"
    config[key] = new
 
for product in all_products:
    key = "PR_" + product
    new = copy.deepcopy(base)
    new["productos"] = [product]
    new["tags"]["type"] = "monoproducto"
    config[key] = new


with open("config/model_stock.json", "w") as config_file: 
    json.dump(config, config_file)

