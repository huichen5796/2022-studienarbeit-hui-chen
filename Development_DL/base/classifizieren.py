import requests

## https://app.nanonets.com/#/
## Use the transfer model to build my classifier for classifying graphs and text.


'''
url = 'https://app.nanonets.com/api/v2/ImageCategorization/LabelUrls/'

headers = {
  'accept': 'application/x-www-form-urlencoded'
}

data = {'modelId': '270402d6-42f7-497b-abce-d897e89b8a81', 'urls' : ['https://goo.gl/ICoiHc']}

response = requests.request('POST', url, headers=headers, auth=requests.auth.HTTPBasicAuth('dipIx95yLD_Ix6RGClR4LEO1KuFBk_hR', ''), data=data)

print(response.text)

'''



url = 'https://app.nanonets.com/api/v2/ImageCategorization/LabelFile/'

data = {'file': open('Development_DL\\Arbeitbreich_DL\\textandtablewinkel.png', 'rb'), 'modelId': ('', '270402d6-42f7-497b-abce-d897e89b8a81')}

response = requests.post(url, auth= requests.auth.HTTPBasicAuth('dipIx95yLD_Ix6RGClR4LEO1KuFBk_hR', ''), files=data)

print(eval(response.text).get('result'))