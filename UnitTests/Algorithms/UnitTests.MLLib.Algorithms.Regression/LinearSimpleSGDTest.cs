using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.Regression;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.Portable.Core;

namespace UnitTests_Regression
{
    [TestClass]
    public class LinearSimpleSGDTest
    {
        double[][] _trainingData;
        string[] _attributeHeaders;
        int _indexTargetAttribute;

        private void initData()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                    "Y"};
            _indexTargetAttribute = 1;
        }


        private void initData_Power()
        {
            initData();
            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 2, 3, 4, 5 };
            _trainingData[1] = new double[] { 1, 4, 9, 16, 25 };
        }

        //Initialize data from Jason's book
        private void initData_Jason()
        {
            initData();
            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 2, 4, 3, 5 };
            _trainingData[1] = new double[] { 1, 3, 3, 2, 5 };
        }


        [TestMethod]
        public void Regression_Linear_gd_check_model_jason_input()
        {
            initData_Jason();
            BuildLinearSGD lm = new BuildLinearSGD();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double[] validateData = { 1 };
            double value = mb.RunModelForSingleData(
                 validateData);

            Assert.IsTrue(value > 1.1 && value < 1.3);
        }

        [TestMethod]
        public void Regression_linear_gd_check_rmse_jason_input()
        {
            initData_Jason();

            BuildLinearSGD lm = new BuildLinearSGD();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb =
                                        (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                                        lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double value = mb.GetModelRMSE(_trainingData);

            Assert.IsTrue(value > .68 && value < .70);
        }


        [TestMethod]
        public void Regression_linear_gd_check_model_power_input()
        {
            initData_Power();

            BuildLinearSGD lm = new BuildLinearSGD();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double[] validateData = { 4 };
            double value = mb.RunModelForSingleData(
                 validateData);

            Assert.IsTrue(value > 17.10 && value < 17.13);
        }

        [TestMethod]
        public void Regression_linear_gd_check_rmse_power_input()
        {
            initData_Power();

            BuildLinearSGD lm = new BuildLinearSGD();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            double value = mb.GetModelRMSE(_trainingData);

            //Some issues were seen earlier maybe with parrelization 
            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 3.18));
        }


        [TestMethod]
        public void Regression_linear_gd_check_rmse_power_param()
        {
            initData_Power();

            BuildLinearSGD lm = new BuildLinearSGD();
            lm.SetParameters(.02, 20);

            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            double value = mb.GetModelRMSE(_trainingData);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value,3.16));
        }


        [TestMethod]
        public void Regression_linear_gd_check_model_power_input_param()
        {
            initData_Power();

            BuildLinearSGD lm = new BuildLinearSGD();
            lm.SetParameters(.005, 30);
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double[] validateData = { 4 };
            double value = mb.RunModelForSingleData(
                 validateData);

            Assert.IsTrue(value > 16.64 && value < 16.66);
        }
    }
}
