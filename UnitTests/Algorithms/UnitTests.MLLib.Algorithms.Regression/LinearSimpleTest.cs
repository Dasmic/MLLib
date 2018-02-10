using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.Regression;
using Dasmic.Portable.Core;

namespace UnitTests_Regression
{
    [TestClass]
    public class LinearSimpleTest:BaseTest
    {
      
        public LinearSimpleTest()
        {
           
        }
        
        
        [TestMethod]
        public void Regression_linear_check_model_jason_input()
        {
            Init_dataset_jason_linear_regression();
            BuildLinearSimple lm = new BuildLinearSimple();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase) lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double[] validateData = { 1 };
            double value = mb.RunModelForSingleData(
                 validateData);

            Assert.IsTrue(value > 1.1 && value < 1.3);
        }


        [TestMethod]
        public void Regression_linear_check_model_jason_input_row_2()
        {
            Init_dataset_jason_linear_regression();
            BuildLinearSimple lm = new BuildLinearSimple();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double[] data = GetSingleTrainingRowDataForTest(2);
            double value = mb.RunModelForSingleData(
                 data);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 3.59));
        }

        [TestMethod]
        public void Regression_linear_check_rmse_jason_input()
        {
            Init_dataset_jason_linear_regression();

            BuildLinearSimple lm = new BuildLinearSimple();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = 
                (Dasmic.MLLib.Algorithms.Regression.ModelBase)lm.BuildModel(_trainingData,
                _attributeHeaders, _indexTargetAttribute);

            double value = mb.GetModelRMSE(_trainingData);

            Assert.IsTrue(value > .68 && value < .70);
        }


        [TestMethod]
        public void Regression_linear_check_model_power_input()
        {
            Init_dataset_power();

            BuildLinearSimple lm = new BuildLinearSimple();

            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData, 
                _attributeHeaders, _indexTargetAttribute);

            double [] validateData = {4};
            double value=mb.RunModelForSingleData(
                 validateData);

            Assert.IsTrue(value > 16.9 && value < 17.1 );
        }

        [TestMethod]
        public void Regression_linear_check_rmse_power_input()
        {
            Init_dataset_power();

            BuildLinearSimple lm = new BuildLinearSimple();
            Dasmic.MLLib.Algorithms.Regression.ModelBase mb = (Dasmic.MLLib.Algorithms.Regression.ModelBase)
                lm.BuildModel(_trainingData,
                _attributeHeaders, 
                _indexTargetAttribute);

            double value = mb.GetModelRMSE(_trainingData);

            Assert.IsTrue(value > 1.66 && value < 1.68);
        }


    }
}
