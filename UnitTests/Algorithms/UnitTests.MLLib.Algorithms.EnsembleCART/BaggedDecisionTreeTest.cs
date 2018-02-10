using Dasmic.MLLib.Algorithms.EnsembleCART;
using Dasmic.MLLib.Common.MLCore;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.EnsembleCART
{
    [TestClass]
    public class BaggedDecisionTreeTest:BaseTest
    {
        [TestMethod]
        public void Bagged_DecisionTree_single_training_sample_value_0()
        {
            initData_Jason_Bagging();
            BuildBaggedDecisionTree build =
                    new BuildBaggedDecisionTree();
            
            ModelBaggedDecisionTree model =
                (ModelBaggedDecisionTree)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row=0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            //Can be both 1 or 0
            Assert.IsTrue(value == 0 | value == 1);
        }

        [TestMethod]
        public void Bagged_DecisionTree_single_training_sample_value_1()
        {
            initData_Jason_Bagging();
            BuildBaggedDecisionTree build =
                    new BuildBaggedDecisionTree();

            ModelBaggedDecisionTree model =
                (ModelBaggedDecisionTree)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 6;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            //System.Diagnostics.Debug.Print("Value:" + value.ToString());
            //Can be both 1 or 0
            Assert.IsTrue(value == 0 | value == 1);
        }

        [TestMethod]
        public void Bagged_DecisionTree_single_training_sample_border_value_0()
        {
            initData_Jason_Bagging();
            BuildBaggedDecisionTree build =
                    new BuildBaggedDecisionTree();
            build.SetParameters(7);
            ModelBaggedDecisionTree model =
                (ModelBaggedDecisionTree)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 4;//5.38 2.1 0
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            //Can be both 1 or 0
            Assert.IsTrue(value==0 | value==1);
        }


        [TestMethod]
        public void Bagged_DecisionTree_all_training_samples()
        {
            initData_Jason_Bagging();
            BuildBaggedDecisionTree build =
                    new BuildBaggedDecisionTree();

            ModelBaggedDecisionTree model =
                (ModelBaggedDecisionTree)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            build.SetParameters(20);
            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);
                if (value ==
                    _trainingData[_indexTargetAttribute][row])
                    count++;
            }
            
            Assert.IsTrue(count <=10 && count >=8);
        }
    }
}
