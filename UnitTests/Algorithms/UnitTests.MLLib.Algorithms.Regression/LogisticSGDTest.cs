using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.Regression;
using Dasmic.MLLib.Common.MLCore;

namespace UnitTests_Regression
{
    [TestClass]
    public class LogisticSGDTest
    {
        double[][] _trainingData;
        string[] _attributeHeaders;
        int _indexTargetAttribute;

        private void initData()
        {
            _attributeHeaders = new string[] {
                                     "X1", "X2",
                                    "Y"};
            _indexTargetAttribute = 2;
        }

        //Initialize data from Jason's book
        private void initData_Jason()
        {
            initData();
            _trainingData = new double[3][];
          
            _trainingData[0] = new double[] { 2.7810836,1.465489372,3.396561688,1.38807019,
                                3.06407232,7.627531214,5.332441248,6.922596716,
                                8.675418651,7.673756466};

            _trainingData[1] = new double[] {2.550537003,2.362125076, 4.400293529,1.850220317,
                                                3.005305973,2.759262235,2.088626775,
                                                1.77106367,-.242068655,3.508563011 };
 
            _trainingData[2] = new double[] { 0,0,0,0,0, 1, 1, 1, 1,1 };
        }


        [TestMethod]
        public void Regression_logistic_gd_check_model_jason_input()
        {
            initData_Jason();
            BuildLogisticSGD lm = new BuildLogisticSGD();

            Dasmic.MLLib.Algorithms.Regression.ModelBase mb= (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double[] validateData = { 1.465489372, 2.362125076 };
            double value = mb.RunModelForSingleData(
                 validateData);
            Assert.IsTrue(value > .13 && value < .15);

            validateData[0] = 7.673756466;
            validateData[1] = 3.508563011;
            value = mb.RunModelForSingleData(
                 validateData);
            Assert.IsTrue(value > .8 && value < .99);
        }


        [TestMethod]
        public void Regression_logistic_gd_check_accuracy_model_jason_input()
        {
            initData_Jason();
            BuildLogisticSGD lm = new BuildLogisticSGD();

            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double [] validateData = new double[2];
            double crisp;
            for(int idx =0;idx < _trainingData[0].Length;idx++)
            {
                validateData[0] = _trainingData[0][idx]; 
                validateData[1] = _trainingData[1][idx];       
                double value = mb.RunModelForSingleData(
                                    validateData);

                if (value < .5) crisp = 0;
                else crisp = 1;

                Assert.AreEqual(crisp, _trainingData[_indexTargetAttribute][idx]);
            }
        }
    }
}
 
