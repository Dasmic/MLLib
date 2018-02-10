using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DecisionTree;
using Dasmic.Portable.Core;
using Dasmic.MLLib.UnitTest;

namespace UnitTests_DecisionTree
{
    [TestClass]
    public class CARTTest : Base
    {
        public CARTTest()
        {
            
        }

      

        [TestMethod]
        public void CART_maketree_check_root_node_structure()
        {
            initData_Jason();
            BuildCART cart =
                    new BuildCART();
            cart.SetMissingValue(999);
            cart.SetParameters(1);

            ModelBase mb =
                (ModelBase)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            Assert.AreEqual(mb.Root.Children.Count, 2);
            Assert.AreEqual(mb.Root.Children[0].Children, null);
            Assert.AreEqual(mb.Root.Children[1].Children, null);
        }

        [TestMethod]
        public void CART_maketree_validate_single_training_data()
        {
            initData_Jason();
            BuildCART cart =
                    new BuildCART();
            cart.SetMissingValue(999);
            cart.SetParameters(1);

            ModelCART mb =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);
            double[] data = new double[] { _trainingData[0][0], _trainingData[1][0] };
            double value = mb.RunModelForSingleData(data);  
            Assert.AreEqual(value, _trainingData[2][0]);
        }


        [TestMethod]
        public void CART_maketree_check_all_training_data_set()
        {
            initData_Jason();
            BuildCART cart =
                    new BuildCART();
            cart.SetMissingValue(999);
            cart.SetParameters(1);

            ModelCART mb =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            for(int row=0;row<_trainingData[0].Length;row++)
            {
                double[] data = new double[] { _trainingData[0][row], _trainingData[1][row] };
                double value = mb.RunModelForSingleData(data);
                Assert.AreEqual(value, _trainingData[2][row]);            
            }
        }


        [TestMethod]
        public void CART_maketree_check_all_validation_data_set()
        {
            initData_Jason();
            BuildCART cart =
                    new BuildCART();
            cart.SetMissingValue(999);
            cart.SetParameters(1);

            ModelCART mb =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int count=0;
            for (int row = 0; row < _validationData[0].Length; row++)
            {
                double[] data = new double[] { _validationData[0][row], _validationData[1][row] };
                double value = mb.RunModelForSingleData(data);
                if (value == _validationData[2][row])
                    count++;
            }
            Assert.AreEqual(count, 9);
        }

        [TestMethod]
        public void CART_maketree_check_root_node_value()
        {
            initData_Jason();
            BuildCART cart =
                    new BuildCART();
            cart.SetMissingValue(999);
            cart.SetParameters(1);

            ModelCART mb =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

             Assert.AreEqual(mb.Root.AttributeIndex, 0);
             Assert.AreEqual(mb.Root.Value, 6.642287351);
            
        }


        [TestMethod]
        public void CART_gini_index_test()
        {
            initData_Jason();
            BuildCART cart =
                    new BuildCART();

            setPrivateVariablesInBuildObject(cart);

            PrivateObject obj = new PrivateObject(cart);
            double gini = (double)obj.Invoke("GetGiniImpurity",
                                            new object[] {
                                                0,
                                                2.771244718 });
            Assert.IsTrue(
                SupportFunctions.DoubleCompare(gini, 0.49382716));
        }


        [TestMethod]
        public void CART_maketree_four_samples()
        {
            initData_four_samples();
            BuildCART cart =
                    new BuildCART();
            ModelCART model =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void CART_maketree_same_four_samples()
        {
            initData_same_four_samples();
            BuildCART cart =
                    new BuildCART();
            ModelCART model =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }


        /// <summary>
        /// R: library(rpart)
        /// input.data<-read.csv("C:\\Users\\Chaitanya Belwal\\Desktop\\data.csv")
        /// fit<-rpart(Y ~ X1+X2, data=input.data, method='class')
        /// print(fit)
        /// </summary>
        [TestMethod]
        public void CART_maketree_special_no_splitting_possible_false()
        {
            initData_special_no_splitting_possible();
            BuildCART cart =
                    new BuildCART();
            ModelCART model =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            Assert.AreNotEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void CART_maketree_special_no_splitting_possible_true()
        {
            initData_special_no_splitting_possible();
            BuildCART cart =
                    new BuildCART();
            ModelCART model =
                (ModelCART)cart.BuildModel(_trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 1;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }
        //Need tests for CART tree with more depth
    }
}
