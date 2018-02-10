using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.Regression;
using Dasmic.Portable.Core;

namespace UnitTests_Regression
{
    [TestClass]
    public class LinearMultiVariableTest:BaseTest
    {
        [TestMethod]
        public void Regression_linear_multivariable_web_example_single_row_0()
        {
            InitData_dataset_regression_web_example();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();
            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)lm.BuildModel(_trainingData,
                                    _attributeHeaders, _indexTargetAttribute);            
            double[] data = GetSingleTrainingRowDataForTest(0);
            double value = mb.RunModelForSingleData(data);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 249.98));            
        }

        [TestMethod]
        public void Regression_linear_multivariable_web_example_single_row_17()
        {
            InitData_dataset_regression_web_example();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();

            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)lm.BuildModel(_trainingData,
                                    _attributeHeaders, _indexTargetAttribute);

            double[] data = GetSingleTrainingRowDataForTest(16);
            double value = mb.RunModelForSingleData(data);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 340.37));
        }

        [TestMethod]
        public void Regression_linear_multivariable_jason_simple()
        {
            Init_dataset_jason_linear_regression();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();

            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)lm.BuildModel(_trainingData,
                                    _attributeHeaders, _indexTargetAttribute);

            double[] data = { 1 };
            double value = mb.RunModelForSingleData(data);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 1.2));
        }

        [TestMethod]
        public void Regression_linear_multivariable_jason_simple_row_2()
        {
            Init_dataset_jason_linear_regression();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();

            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)lm.BuildModel(_trainingData,
                                    _attributeHeaders, _indexTargetAttribute);

            double[] data = GetSingleTrainingRowDataForTest(2);
            double value = mb.RunModelForSingleData(data);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 3.60));
        }

        [TestMethod]
        public void Regression_linear_multivariable_jason_simple_rmse()
        {
            Init_dataset_jason_linear_regression();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();

            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)lm.BuildModel(_trainingData,
                                    _attributeHeaders, _indexTargetAttribute);

            //Refill the trainingData array since it gets messed up
            Init_dataset_jason_linear_regression();
            double value = mb.GetModelRMSE(_trainingData);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value,
                                    0.69));
        }


        [TestMethod]
        public void Regression_linear_multivariable_pythagoras_row_5()
        {
            Init_dataset_pythagoras();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();

            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)
                                     lm.BuildModel(
                                    _trainingData,
                                    _attributeHeaders,
                                    _indexTargetAttribute);

            
            int row = 5;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = mb.RunModelForSingleData(data);
            
            //Value is within +-1.0
            Assert.IsTrue(value < 
                                _trainingData[_indexTargetAttribute][row]+1.0
                            && value > _trainingData[_indexTargetAttribute][row]-1.0);
        }

        [TestMethod]
        public void Regression_linear_multivariable_pythagoras_rmse()
        {
            Init_dataset_pythagoras();
            BuildLinearMultiVariable lm = new BuildLinearMultiVariable();

            ModelLinearMultiVariableBase mb = (ModelLinearMultiVariableBase)
                                     lm.BuildModel(
                                    _trainingData,
                                    _attributeHeaders, 
                                    _indexTargetAttribute);

            //Refill the trainingData array since it gets messed up            
            double value = mb.GetModelRMSE(_trainingData);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value,
                                    0.083));
        }

    }
}
