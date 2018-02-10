using System;
using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Math.Statistics;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Linq;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public abstract class BuildBase : Common.MLCore.BuildBase
    {
        protected int _minimumNumberPerNode;
        double _confidenceFactor;

        public BuildBase()
        {
             _minimumNumberPerNode = 1;
        }

        //Configuration Properties
        public int MinimumNumberPerNode
        {
            get { return _minimumNumberPerNode; }
            private set {
                if (value < 1)
                    _minimumNumberPerNode = 1;
                else
                    _minimumNumberPerNode = value; }
        }

        public double ConfidenceFactor
        {
            get { return _confidenceFactor; }
            private set { _confidenceFactor = value; }
        }

        /// <summary>
        /// Sets parameters
        /// MinimumNumberPerNode, Confidence Factor
        /// </summary>
        /// <param name="values"></param>
        public override void
              SetParameters(params double[] values)
        {
            if(values.Length > 0) MinimumNumberPerNode = (int)values[0];
            if (values.Length > 1) ConfidenceFactor = values[1];
        }

        //Make private member so that stack is not full
        protected virtual DecisionTreeNode buildChildNodes(ConcurrentBag<long> trainingDataRowIndices,
                                                   double value,
                                                   DecisionTreeNode dtnParent)
        {
            DecisionTreeNode dtn = new DecisionTreeNode(dtnParent);

            //Get all rows in Training Data
            FilteredData fd = getFilteredDataForNode(dtn, value,
                                             trainingDataRowIndices);

            //Check if all target examples are same or not
            //Their Entropy will be 0

            if (fd.NumberOfRows <= _minimumNumberPerNode || isTargetDataSame(fd.FilteredDataValues)) //Attributes is empty
            {
                setAsTargetAttributeNode(fd.FilteredDataValues, dtn);
                return dtn;//No more children if attributeIndex is target Attributes
            }

            //Set data for current node
            SplittedAttributeData ed =
                    splitDataOnUnivariateCriterion(fd.FilteredDataValues);

            //Check for positive and negative examples
            dtn.setAttributeValues(ed.AttributeIndex,
                            _attributeHeaders[ed.AttributeIndex]);

            ConcurrentBag<long> newTrainingDataRowIndices =
                                fd.TrainingDataRowIndices;
            fd = null;  //Free Memory -> Clean up data, no longer needed
            //Key has values
            foreach (double key in ed.Freqs.Keys)
            {
                if (key != 999) //Dont add for missing values
                    dtn.addChild(key, buildChildNodes(newTrainingDataRowIndices,
                                                    key,
                                                    dtn)); //Key has value
            }
            return dtn;
        }
    

        protected void setAsTargetAttributeNode(double[][] filteredValues,
                                DecisionTreeNode dtn)
        {
            /*if (filteredValues[0].Length < 1)
            {
                dtn.setAttributeValues(-1, "");//Denotes and invalid node                                
            }
            else*/
            {
                dtn.setAttributeValues(_indexTargetAttribute,
                                _attributeHeaders[_indexTargetAttribute]);
                dtn.Value = getMostFrequentValueForIndex(filteredValues, _indexTargetAttribute);
            }
            return;
        }

        protected bool isTargetDataSame(double[][] filteredData)
        {
            double entropy = InformationGain.EntropyShannon(filteredData[_indexTargetAttribute]);
            if (entropy == 0) return true;
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filteredData"></param>
        /// <returns>HighestEntropyData object. Will be null if TargetAttribute</returns>
        protected virtual double
                getMostFrequentValueForIndex(
                        double[][] filteredData, int index)
        {
            return getMostFrequentValueForIndex_Generic(
                  filteredData, index, false);
        }

        /// <summary>
        /// Return the target attribute values based on value of another index
        /// </summary>
        /// <param name="data"></param>
        /// <param name="attributeIdx"></param>
        /// <param name="attributeValue"></param>
        /// <returns></returns>
        protected double[] getFilteredTargetValues(double[][] data,
                                                 int attributeIdx,
                                                 double attributeValue)
        {
            List<double> filteredValues = new List<double>();
            for (int rowIdx = 0;
                rowIdx < data[_indexTargetAttribute].Length; rowIdx++)
            {
                if (data[attributeIdx][rowIdx] == attributeValue)
                    filteredValues.Add(data[_indexTargetAttribute][rowIdx]);
            }
            return filteredValues.ToArray();
        }

        //
        //Used original data

        /// <summary>
        /// Return filtered data for node
        /// 
        /// Returns rows which match parent attribute having parent value
        /// Also sets the attribute column of all parents to null
        /// 
        /// </summary>
        /// <param name="dtn"></param>
        /// <param name="nodeValue"></param>
        /// <param name="trainingDataRowIndices"></param>
        /// <returns></returns>
        protected virtual FilteredData getFilteredDataForNode(
                                                DecisionTreeNode dtn,
                                                      double nodeValue,
                                                      ConcurrentBag<long> trainingDataRowIndices)
        {
            //filterOnValues has all values we need to filter on
            if (dtn.Parent == null) //Root nodes, gets entire training data
            {
                return new FilteredData(_trainingData,
                                    trainingDataRowIndices, 
                                    trainingDataRowIndices.Count);
            }

            Dictionary<int, double> allParentValues =
                                    new Dictionary<int, double>();
            DecisionTreeNode parent = dtn.Parent;

            
            while (parent != null) //Find out values for each parent
            {
                allParentValues.Add(parent.AttributeIndex,
                                        nodeValue); //Value is in current node
                parent = parent.Parent;
            } //Only add 1 index, value pair since trainingDataRowIndices is already sorted out

            ConcurrentBag<long> bagTrainingDataRowIndices = 
                new ConcurrentBag<long>();

            
            // Find out which rows matach the parent
            Parallel.ForEach(trainingDataRowIndices, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },rowIdx =>
             {               
                 if(_trainingData[dtn.Parent.AttributeIndex][rowIdx] ==
                                 nodeValue)
                 {
                     bagTrainingDataRowIndices.Add(rowIdx);
                 }
             }); //rowIdx

            //Now rows are known allocate for filtered Value
            //Allocate for filteredValues
            //Need dymanic numer of rows hence using fltered values
            long[] arrayTrainingDataRowIndices = bagTrainingDataRowIndices.ToArray<long>();
            double[][] filteredData = new double[_trainingData.Length][];
            //Parallel.For(0,_trainingData.Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, colIdx =>           
            for(int colIdx=0; colIdx < _trainingData.Length; colIdx++)
            {
                    filteredData[colIdx] = new double[arrayTrainingDataRowIndices.Length];
                    long rowIdx;
                    //Will run in parallel Order has to be presevered
                    for (int idx = 0; idx < arrayTrainingDataRowIndices.Length; idx++)
                    {
                        rowIdx = arrayTrainingDataRowIndices[idx];
                        filteredData[colIdx][idx] = 
                            _trainingData[colIdx][rowIdx];
                    }
            }//);
      
            return new FilteredData(filteredData,
                bagTrainingDataRowIndices,
                filteredData[0].Length);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="attributeIdx"></param>
        /// <param name="attributeValue"></param>
        /// <returns></returns>
        protected double getEntropyOfTargetAttribute(double[][] data)
        {
            return InformationGain.EntropyShannon(
                    data[_indexTargetAttribute]);
        }

        #region Overridden Routines

        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute);


        #endregion


        #region Abstract Routines
        protected abstract SplittedAttributeData
               splitDataOnUnivariateCriterion(
                       double[][] data);

        
        


        #endregion

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filteredData"></param>
        /// <returns>HighestEntropyData object. Will be null if TargetAttributes</returns>
        protected virtual double
                getMostFrequentValueForIndex_Generic(
                        double[][] filteredData,
                        int index,
                        Boolean checkMissingValue)
        {
            Dictionary<double, long> freqs;
            freqs = InformationGain.Frequency(
                                filteredData[index]);

            //Do not count missing values
            if(checkMissingValue)
                if (freqs.ContainsKey(_missingValue))
                    freqs.Remove(_missingValue);

            //Aggregate is the LINQ name for the commonly known functional concept Fold
            var max = freqs.Aggregate((l, r) => l.Value >
                            r.Value ? l : r).Key;
            return max;
        }

        /// <summary>
        /// Function used only for use in bagging
        /// when rows in training data are passed.
        /// This builds the initial data structures
        /// </summary>
        /// <param name="trainingDataRowIndices"></param>
        /// <returns></returns>
        protected FilteredData convertRowIndicesToFilteredData(ConcurrentBag<long> trainingDataRowIndices)
        {
            double[][] filteredData = Get2DArray(_trainingData.Length, trainingDataRowIndices.Count);
            long idx = 0;
            foreach (long rowIdx in trainingDataRowIndices)
            //Parallel.ForEach(trainingDataRowIndices, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, rowIdx =>
            {
                for (int colIdx = 0; colIdx < _trainingData.Length; colIdx++)
                {
                    filteredData[colIdx][idx] = _trainingData[colIdx][rowIdx];
                }
                idx++;
            }
            return new FilteredData(filteredData,
                       trainingDataRowIndices,
                       trainingDataRowIndices.Count);
        }
    }
}

