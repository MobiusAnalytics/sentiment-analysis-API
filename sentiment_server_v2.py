from email import message
from flask import Flask, jsonify,request
from flask_restful import Resource, Api
import traceback
from datetime import datetime
import EducationModels
import EcommerceModels
import pandas as pd

# Initilizing the app and API
app = Flask(__name__)
api = Api(app)

#For Error Logging
def LogError(message):
    with open("ErrorLog.log", "a") as efr:
        efr.write("ERROR :"+datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p"))
    traceback.print_tb(message.__traceback__, file=open("ErrorLog.log", "a"))
    return jsonify({'status': 'Error','message':'Internal Server Error'})
        
#For API design
class Test(Resource):
    def get(self):
        try:
            return jsonify({'status': 'ok','message':'get'})
        except Exception as e:
            return LogError(e)


class Education(Resource):
    def appendCSV(self,data):
        try:
            data.to_csv('userReview.csv', mode='a', index=False, header=False)
        except:
            pass

    def convert2JSON(self,data):
        jsonData = eval(data.to_json(orient = 'index'))
        return {'status':'ok','message':jsonData}

    def get(self):
        Test.get()

    def post(self):
        try:
            review = request.get_json()
            ugc=review['ugc']
            if ugc=="":
                return jsonify({'status':'ok','message':'Empty review'})
            categoryId=review['categoryId']
            data=pd.DataFrame({"ugc":[ugc]})
            if categoryId=="60":
                sentiment=EcommerceModels.sentimentAnalyzer(data)
            elif categoryId=="61":
                sentiment=EducationModels.sentimentAnalyzer(data)
            else:
                return jsonify({'status':'ok','message':'No Suitable model for the category ID'})
            sentimentJson=self.convert2JSON(sentiment)
            self.appendCSV(sentiment)
            return jsonify(sentimentJson)
        except Exception as e:
            return LogError(e)


#routing the API
api.add_resource(Test, '/')
api.add_resource(Education,'/api/v1/sentimentMaster')

if __name__ == '__main__':
    app.run(debug = True)
