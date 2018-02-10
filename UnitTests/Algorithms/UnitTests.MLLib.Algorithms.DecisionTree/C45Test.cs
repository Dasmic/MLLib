using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DecisionTree;

namespace UnitTests_DecisionTree
{
    [TestClass]
    public class C45Test:Base
    {
        
        public  C45Test()
        {
            initData_Mitchell_Book_Missingdata();
        }

        
        /// <summary>
        /// R:
        /// library(party)
        /// output.tree <- ctree(PlayTennis ~ Outlook+Temperature+Humidity+Wind, data = input.data)
        /// plot(output.tree)
        /// NOTE: The tree in R is not correct
        /// </summary>
        [TestMethod]
        public void C45_maketree_check_final_node_values()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase)c45.BuildModel(_trainingData, 
                _attributeHeaders,
                _indexTargetAttribute);
            model.getPrintedTree();

            //Check Tree Nodes
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(2).getChildWithValue(0).Value==1); //Wind
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(2).getChildWithValue(1).Value==0);
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(1).Value == 1);
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(0).Value == 1);
            
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(0).Value == 0); //Outlook
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(1).Value == 1);
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(2).Value == 0);

        }

        /// <summary>
        /// R:
        /// library(party)
        /// output.tree <- ctree(PlayTennis ~ Outlook+Temperature+Humidity+Wind, data = input.data)
        /// plot(output.tree)
        /// NOTE: The tree in R is not correct
        /// </summary>
        [TestMethod]
        public void C45_maketree_check_level1_node()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase)c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);
            model.getPrintedTree();

            //Check Tree Nodes
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(0).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(2).AttributeName.Equals("Wind"));
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(1).AttributeName.Equals("PlayTennis"));
            
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(0).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(1).AttributeName.Equals("PlayTennis"));
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(2).AttributeName.Equals("PlayTennis"));
        }


        /// <summary>
        /// R:
        /// library(party)
        /// output.tree <- ctree(PlayTennis ~ Outlook+Temperature+Humidity+Wind, data = input.data)
        /// plot(output.tree)
        /// NOTE: The tree in R is not correct
        /// </summary>
        [TestMethod]
        public void C45_maketree_check_root_node()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase)c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);
            model.getPrintedTree();

            //Check Nodes
            Assert.IsTrue(model.Root.AttributeName.Equals("Humidity"));
            Assert.IsTrue(model.Root.getChildWithValue(0).AttributeName.Equals("Outlook"));
            Assert.IsTrue(model.Root.getChildWithValue(1).AttributeName.Equals("Outlook"));
        }


        [TestMethod]
        public void C45_maketree_minimum_3_check_final_nodes()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            c45.SetParameters(3);
           

            ModelBase model = (ModelBase)c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(2).Value == 1); //Outlook
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(1).Value == 1);
            Assert.IsTrue(model.Root.getChildWithValue(0).getChildWithValue(0).Value == 1);

            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(0).Value == 0); //Outlook
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(1).Value == 1);
            Assert.IsTrue(model.Root.getChildWithValue(1).getChildWithValue(2).Value == 0);
        }



        [TestMethod]
        public void C45_runmodel_case_1()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase)c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 1, 1, 1, 0 };
            double value = model.RunModelForSingleData(data);
            
            Assert.AreEqual(value, 1);
        }


        [TestMethod]
        public void C45_number_misclassified()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model=(ModelBase) c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            DecisionTreePruning dtp = new DecisionTreePruning();
            int value = dtp.getNumberMisclassified(
                                     _trainingData,
                                     _indexTargetAttribute,
                                    model);

            Assert.AreEqual(value, 1);
        }

        [TestMethod]
        public void C45_runmodel_case_2()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase)c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 0, 1, 0, 0 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 1);
        }

        [TestMethod]
        public void C45_runmodel_case_3()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(-999);
            ModelBase model = (ModelBase) c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);
            double[] data = { 0, 1, 1, 0 };
            double value = model.RunModelForSingleData(data);
           
           
            Assert.AreEqual(value, 0);
        }


        [TestMethod]
        public void C45_runmodel_case_4()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase) c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 2, 1, 1, 0 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 0);
        }

        [TestMethod]
        public void C45_runmodel_case_5()
        {
            BuildC45 c45 = new BuildC45();
            c45.SetMissingValue(999);
            ModelBase model = (ModelBase) c45.BuildModel(_trainingData,
                _attributeHeaders,
                _indexTargetAttribute);

            // Data will be in format "Outlook","Temperature","Humidity","Wind","PlayTennis"
            double[] data = { 2, 1, 1, 1 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 0);
        }
        
    }
}
