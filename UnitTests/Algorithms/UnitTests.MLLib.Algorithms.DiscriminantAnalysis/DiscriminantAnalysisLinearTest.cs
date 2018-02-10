using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DiscriminantAnalysis;
using Dasmic.Portable.Core;

namespace UnitTests_DiscriminantAnalysis
{
    [TestClass]
    public class DiscriminantAnalysisLinearTest:BaseTest
    {
        //Compute the LDA
        [TestMethod]
        public void DA_linear_matrix_example()
        {
            initData_dataset_7_row_2_class_example();

            BuildLinear bl = new BuildLinear();

            ModelLinear ml = (ModelLinear)bl.BuildModel
                                (_trainingData, _attributeHeaders, _indexTargetAttribute);

            double[] data;
            double value;
            data = new double[_trainingData.Length - 1];
            for (int row = 0; row < _trainingData[0].Length; row++)
            { 
                for (int col =0;col<_trainingData.Length-1;col++)       
                {
                    data[col] = 
                        _trainingData[col][row];
                }
                //Run model
                value=ml.RunModelForSingleData(data);
                Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
            }

        }


        [TestMethod]
        public void DA_linear_matrix_jason()
        {
            initData_dataset_40_row_1_jason_example();

            BuildLinear bl = new BuildLinear();

            ModelLinear ml = (ModelLinear)bl.BuildModel
                                (_trainingData, _attributeHeaders, _indexTargetAttribute);

            double[] data;
            double value;
            data = new double[_trainingData.Length - 1];
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                for (int col = 0; col < _trainingData.Length - 1; col++)
                {
                    data[col] =
                        _trainingData[col][row];
                }
                //Run model
                value = ml.RunModelForSingleData(data);
                Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
            }

        }
    }
}
