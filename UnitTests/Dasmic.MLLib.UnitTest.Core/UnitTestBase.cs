using System;
using System.IO;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.IO.Windows;
using Dasmic.MLLib.Common.IO;
using Dasmic.Portable.Core;

namespace Dasmic.MLLib.UnitTest.Core
{
    
    public class UnitTestBase
    {
        protected double[][] _trainingData;
        protected string[] _attributeHeaders;
        protected int _indexTargetAttribute;
        protected double[][] _validationData;

        //This assume assignment to local class values is done
        protected void setPrivateVariablesInBuildObject(BuildBase buildBase)
        {
           
            var prop = buildBase.GetType().GetField("_noOfAttributes", System.Reflection.BindingFlags.NonPublic
           | System.Reflection.BindingFlags.Instance);
            prop.GetValue(buildBase);
            prop.SetValue(buildBase, _trainingData.Length - 1);

            prop = buildBase.GetType().GetField("_trainingData", System.Reflection.BindingFlags.NonPublic
            | System.Reflection.BindingFlags.Instance);
            prop.SetValue(buildBase, _trainingData);

            prop = buildBase.GetType().GetField("_indexTargetAttribute", System.Reflection.BindingFlags.NonPublic
            | System.Reflection.BindingFlags.Instance);
            prop.SetValue(buildBase, _indexTargetAttribute);

            prop = buildBase.GetType().GetField("_noOfDataSamples", System.Reflection.BindingFlags.NonPublic
            | System.Reflection.BindingFlags.Instance);
            prop.SetValue(buildBase, _trainingData[0].Length);

        }


        public double[] GetSingleTrainingRowDataForTest(int row)
        {         
            //Remove these lines after some time
            /*double [] data = new double[_trainingData.Length - 1];
            for (int col = 0; col < data.Length; col++)
            {
                data[col] =
                         _trainingData[col][row];
            }
            */

            double[] data = SupportFunctions.GetLinearArray(_trainingData,
                                    row, _trainingData.Length - 2);
            return data;
        }

        public void LoadFromDataSetFile(string fileName, int maxParallelThreads)
        {
            CSVOperations csv = new CSVOperations();
            FileData fd = csv.Read(fileName, maxParallelThreads);

            //Shallow Copy of Data
            _trainingData = fd.values;// new double[fd.attributeHeaders.Length][];
            _attributeHeaders = fd.attributeHeaders;
            _indexTargetAttribute = fd.attributeHeaders.Length - 1;
        }


        public void LoadFromDataSet(EnumDataSets dataSet, int maxParallelThreads)
        {
            string fileName = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName +
                                    "\\..\\..\\..\\DataSets\\" + GetDataSetFileName(dataSet);
            LoadFromDataSetFile(fileName, maxParallelThreads);
        }

        public String GetDataSetFileName(EnumDataSets dataSet)
        {
            string fileName;
            switch (dataSet)
            {
                case EnumDataSets.Pythagoras:
                    fileName= "pythagoras.csv";
                    break;
                case EnumDataSets.Iris:
                    fileName = "iris.csv";
                    break;
                default:
                    fileName = "pythagoras.csv";
                    break;
            }
            return fileName;
        }
    }
}
