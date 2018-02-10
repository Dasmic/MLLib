using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DecisionTree;

namespace UnitTests_DecisionTree
{
    [TestClass]
    public class ID3Test:Base
    {
       
        public  ID3Test()
        {
            initData_Mitchell_Book();
        }


        /// <summary>
        /// R:
        /// library(party)
        /// output.tree <- ctree(PlayTennis ~ Outlook+Temperature+Humidity+Wind, data = input.data)
        /// plot(output.tree)
        /// NOTE: The tree in R is not correct
        /// </summary>
        [TestMethod]
        public void ID3_maketree_check_final_nodes()
        {
            string output;
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = 
                        new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData, 
                _attributeHeaders,
                _indexTargetAttribute);
            output=model.getPrintedTree();

            //Check Tree Nodes
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(0).Value==1); //Humidity
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(1).Value==0);

            Assert.IsTrue(model.Root.getChildWithValue(1).Value == 1);

            Assert.IsTrue(model.Root.getChildWithValue(2).getChildWithValue(0).Value == 1); //Humidity
            Assert.IsTrue(model.Root.getChildWithValue(2).getChildWithValue(1).Value == 0);
        }

        /// <summary>
        /// R:
        /// library(party)
        /// output.tree <- ctree(PlayTennis ~ Outlook+Temperature+Humidity+Wind, data = input.data)
        /// plot(output.tree)
        /// NOTE: The tree in R is not correctRegression
        /// </summary>
        [TestMethod]
        public void ID3_maketree_check_level_1_node()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);
            model.getPrintedTree();

            //Check Tree Nodes
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(0).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(1).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(1).Children == null);
            Assert.IsTrue(model.Root.getChildWithValue(2).getChildWithValue(0).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(2).getChildWithValue(1).AttributeName.Equals("PlayTennis"));
        }


        /// <summary>
        /// R:
        /// library(party)
        /// output.tree <- ctree(PlayTennis ~ Outlook+Temperature+Humidity+Wind, data = input.data)
        /// plot(output.tree)
        /// NOTE: The tree in R is not correct
        /// </summary>
        [TestMethod]
        public void ID3_maketree_check_root_node()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);
            model.getPrintedTree();

            //Check Nodes
            Assert.IsTrue(model.Root.AttributeName.Equals("Outlook"));
            Assert.IsTrue(model.Root.getChildWithValue(0).AttributeName.Equals("Humidity"));
            Assert.IsTrue(model.Root.getChildWithValue(1).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(2).AttributeName.Equals("Wind"));
        }


       
        [TestMethod]
        public void ID3_runmodel_case_1()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            
            double[] data = { 1, 1, 1, 0 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 1);
        }

        [TestMethod]
        public void ID3_runmodel_case_2()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 0, 1, 0, 0 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 1);
        }

        [TestMethod]
        public void ID3_runmodel_case_3()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 0, 1, 1, 0 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 0);
        }


        [TestMethod]
        public void ID3_runmodel_case_4()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase)id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 2, 1, 1, 0 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 1);
        }

        [TestMethod]
        public void ID3_runmodel_case_5()
        {
            Dasmic.MLLib.Algorithms.DecisionTree.BuildID3 id3 = new Dasmic.MLLib.Algorithms.DecisionTree.BuildID3();
            ModelBase model = (ModelBase) id3.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 2, 1, 1, 1 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 0);
        }

    }
}
