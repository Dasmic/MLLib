using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class BuildCART:BuildBase
    {
        protected bool _trainingRowsPassed;
        protected long _numberOfFeaturesForSplit;
        protected int[] _splitFeatures;

        public BuildCART()
        {
            _numberOfFeaturesForSplit = long.MaxValue;
        }

        /// <summary>
        /// Use only when using CART for Random Forest
        /// </summary>
        /// <param name="numberOfFeaturesForSplit"></param>
        public void SetParametersForRandomForest(long numberOfFeaturesForSplit)
        {
            _numberOfFeaturesForSplit = numberOfFeaturesForSplit;
        }
      
        protected Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute,
                             ConcurrentBag<long> trainingDataRowIndices,
                             bool trainingRowsPassed)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelCART mb = new ModelCART(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);
            _trainingRowsPassed = trainingRowsPassed; //Flag to indicate training rows passed, so we will have to build filteredData - for bagging purposes
            //Data is good, proceed
            //trainingDataRowIndices is used to track tree values
            mb.Root = BuildChildNodes(trainingDataRowIndices, 0, null, false);
            return mb;
        }

        public virtual Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute,
                             ConcurrentBag<long> trainingDataRowIndices)
        {
            return BuildModel(trainingData,
                            attributeHeaders,
                            indexTargetAttribute,
                            trainingDataRowIndices, true);
        }


        /// <summary>
        /// Values format is Col, Row
        /// </summary>
        /// <param name="values"></param>
        /// <param name="headers"></param>
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //Set row indices before recursion
            ConcurrentBag<long> trainingDataRowIndices =
                        GetTrainingDataForAllRows(trainingData);
            
            return BuildModel(trainingData,
                             attributeHeaders,
                             indexTargetAttribute,
                             trainingDataRowIndices,false) ;
        }


        protected ConcurrentBag<long> 
                    GetTrainingDataForAllRows(double[][] trainingData)
        {
            ConcurrentBag<long> trainingDataRowIndices =
                        new ConcurrentBag<long>();

            for (long idx = 0; idx < trainingData[0].Length; idx++)
                trainingDataRowIndices.Add(idx);

            return trainingDataRowIndices;
        }




        //Make private member so that stack is not full
        protected DecisionTreeNode BuildChildNodes(ConcurrentBag<long> trainingDataRowIndices,
                                                   double value,
                                                   DecisionTreeNode dtnParent,
                                                   bool isLessThan)
        {
            DecisionTreeNode dtn = new DecisionTreeNode(dtnParent);

            //Get all rows in Training Data
            FilteredData fd = GetFilteredDataForNode(dtn, value,
                                             trainingDataRowIndices,
                                             isLessThan);

            //Stopping Criterion
            //Check if minimum number of nodes are there
            //OR all target values are same
             if (fd.NumberOfRows <= _minimumNumberPerNode
                    || isTargetDataSame(fd.FilteredDataValues) //Attributes is empty
                    || (dtnParent != null  && fd.TrainingDataRowIndices.Count == trainingDataRowIndices.Count)//implies no split happened)
                    || GetAdditionalStoppingCondition(dtn))  
                {
                    if (fd.NumberOfRows == 0) //Special case, use original data as node
                    {
                        fd = convertRowIndicesToFilteredData(trainingDataRowIndices);                    
                    }               
                    setAsTargetAttributeNode(fd.FilteredDataValues, dtn);
                    return dtn;//No more children if min attributes reached
            }

            //Set data for current node
            SplittedAttributeData ed =
                    splitDataOnUnivariateCriterion(fd.FilteredDataValues);

            //Check for positive and negative examples
            dtn.setAttributeValues(ed.AttributeIndex,
                            _attributeHeaders[ed.AttributeIndex]);

            //Store value in column ed.AttributeIndex based on which split  was done
            dtn.Value = ed.SplittingCriteriaValue; 

            ConcurrentBag<long> newTrainingDataRowIndices =
                                fd.TrainingDataRowIndices;
            fd = null;  //Free Memory -> Clean up data, no longer needed
                        //Key has values

            //Add left node
            if (ed.SplittingCriteriaValue != _missingValue) //Dont add for missing values
            {
                //0 if for left, 1 is for right.
                //There wont be any conflict since each node will have only 2 children
                //DecisionTreeNode dtnChild =                
                dtn.addChild(0, BuildChildNodes(newTrainingDataRowIndices,
                                               ed.SplittingCriteriaValue,
                                                dtn, true)); //Key has value

                //dtnChild = 
                dtn.addChild(1, BuildChildNodes(newTrainingDataRowIndices,
                                             ed.SplittingCriteriaValue,
                                              dtn, false)); //Key has value
            }

            return dtn;
        }

        //Any extra stopping condition, primarily added for boosting
        //which needs decision stumps
        protected virtual bool GetAdditionalStoppingCondition(DecisionTreeNode dtn)
        {
            return false;//Make sure value is false else stopping condition is reach automatically
        }

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
        protected FilteredData GetFilteredDataForNode(
                                                DecisionTreeNode dtn,
                                                      double nodeValue,
                                                      ConcurrentBag<long> trainingDataRowIndices,
                                                      bool isLessThan)
        {            
            if (dtn.Parent == null) //Root nodes, gets entire training data
            {
                //If trainingDataRowIndices not same as original rows
                //Then filter, required for Bagged Trees                 
                if (_trainingRowsPassed)
                {
                    return convertRowIndicesToFilteredData(trainingDataRowIndices);
                }
                else
                {
                    return new FilteredData(_trainingData,
                                        trainingDataRowIndices,
                                        trainingDataRowIndices.Count);
                }
            }

            Dictionary<int, double> allParentValues =
                                    new Dictionary<int, double>();
            //Allocate for filteredValues
            List<double>[] filteredValues = new List<double>[_trainingData.Length];
            //filterOnValues has all values we need to filter on
            //Need dynamic numer of rows hence using filtered values as List
            Parallel.For(0, _trainingData.Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },colIdx => {
                filteredValues[colIdx] = new List<double>();
            });

            DecisionTreeNode parent = dtn.Parent;            
            ConcurrentBag<long> newTrainingDataRowIndices =
                new ConcurrentBag<long>();

            //Parallelize this later
            //foreach (long rowIdx in trainingDataRowIndices) //Can optimize here to view rows passed down
            Parallel.ForEach(trainingDataRowIndices, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, rowIdx =>
            {                
                if ((isLessThan && _trainingData[dtn.Parent.AttributeIndex][rowIdx] <
                                nodeValue) || (!isLessThan  && _trainingData[dtn.Parent.AttributeIndex][rowIdx] >=
                                nodeValue))
                {
                    newTrainingDataRowIndices.Add(rowIdx);
                    //Copy array and break from foreach
                    //List.Add is not thread safe
                    CopyFilteredData(filteredValues, rowIdx);                 
                }
            }); //rowIdx

            //Convert List to Array
            //Even copy previous columns, as there Entropy will be 0
            double[][] filteredData = new double[_trainingData.Length][];
            long numberOfRows = 0;
            //for(int colIdx=0; colIdx < _trainingData.Length; colIdx++)
            Parallel.For(0, _trainingData.Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, colIdx =>
            {
                    //No need to set parent to NULL in CART
                    filteredData[colIdx] =
                        filteredValues[colIdx].ToArray();
                    //Do note move this out or for since  filteredData[colIdx] can be null
                    //numberOfRows is same for all selected columns
                    numberOfRows = filteredData[colIdx].Length;
            });
            return new FilteredData(filteredData,
                newTrainingDataRowIndices,
                numberOfRows);
        }

        

        protected void CopyFilteredData(List<double>[] filteredValues,long rowIdx)
        {
            for (int colIdx = 0; colIdx < _trainingData.Length; colIdx++)
            {
                //Order has to be presevered
                filteredValues[colIdx].Add(
                    _trainingData[colIdx][rowIdx]);
            };
        }
        

        protected int[] GetFeaturesForSplitCritertion()
        {                    
            if (_numberOfFeaturesForSplit == long.MaxValue ||
                _numberOfFeaturesForSplit == _trainingData.Length - 1)
            {
                //Use _splitFeatures if Feature index is same as all number of features(non-Random Forest tree) 
                //This avoids repetitive computations
                if (_splitFeatures == null)
                {
                    _splitFeatures = new int[_trainingData.Length - 1];
                    for (int ii =0;ii<_splitFeatures.Length;ii++)
                    {
                        _splitFeatures[ii] = ii;
                    }
                }
                return _splitFeatures; 
            }
            else //Is for random forest, general random split points
            {
                int[] features;
                features = GetUniqueRandomNumbers(0, 
                            _trainingData.Length - 1,
                            (int) _numberOfFeaturesForSplit);
                return features;
            }
        }

        /// <summary>
        /// Split data based on Gini Index
        /// 
        /// ed.SplittingCriteriaValue contains value of ed.Index
        /// </summary>
        /// <param name="data">Subset of Training Data</param>
        /// <returns>HighestEntropyData object. Will be null if TargetAttributes</returns>
        protected override SplittedAttributeData
                splitDataOnUnivariateCriterion(
                        double[][] data)
        {
            int[] features = GetFeaturesForSplitCritertion();

            //Store values for each col to help in parallelization
            double[] minGini = new double[features.Length - 1];
            double[] splitCriterion = new double[features.Length - 1];
            int[] attrIdx = new int[features.Length - 1];

            //for (int col = 0; col < data.Count()-1;col++)
            //Assume last column is for value, d not compute gini split for target value
            Parallel.For(0, features.Count() - 1, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, col =>
                 {
                     double gini;
                     minGini[col] = double.PositiveInfinity;
                     for (int row = 0; row < data[0].Count(); row++)
                     {
                      //gini = getGiniImpurity(col, data[col][row]);
                      gini = GetGiniImpurity(data, row, col);

                      //Compute Gini  
                      if (gini < minGini[col])
                         {
                             attrIdx[col] = col;
                             splitCriterion[col] = data[col][row];
                             minGini[col] = gini;
                         }

                         if (minGini[col] == 0)//Perfect split point found, no need to continue further
                          break;//return ed; 
                  }//If condition
              }); //Main loop

            //Find value with lowest minGini
            int minIdx = 0;
            for (int col = 1; col < minGini.Length; col++)
            {
                if (minGini[col] < minGini[minIdx])
                    minIdx = col;
            }

            SplittedAttributeData ed = new SplittedAttributeData()
            {
                AttributeIndex = attrIdx[minIdx],
                SplittingCriteriaValue = splitCriterion[minIdx]
            };
            
            return ed;
        }


        protected virtual double GetGiniImpurity(double [][] data,
                                            int row,
                                            int col)
        {
            return GetGiniImpurity(col, data[col][row]);
        }

        /// <summary>
        /// Use the < and >= conditions
        /// 
        /// Assumes trainingData last column is attribute index
        /// </summary>
        /// <returns></returns>
        protected virtual double GetGiniImpurity(int splitValueIndex, 
                                            double splitValue)
        {
            double giniImpurity = 0;
            double targetValue=0;
            ConcurrentDictionary<double, double> left = new ConcurrentDictionary<double, double>();
            ConcurrentDictionary<double, double> right = new ConcurrentDictionary<double, double>();
            ConcurrentDictionary<double, double> allClasses = new ConcurrentDictionary<double, double>();

            //Partition on trainingData[splitIndex][] < splitValue
            //and trainingData[splitIndex][] >= splitValue 
            for (int rowIdx=0;rowIdx < _trainingData[0].Length;
                                rowIdx++)
            {
                targetValue = _trainingData[_indexTargetAttribute][rowIdx];
                allClasses.AddOrUpdate(targetValue, 1, (key, oldValue) => oldValue+1);

                if (_trainingData[splitValueIndex][rowIdx] < splitValue)
                {
                    //Add element to left such that value is the total number of elements
                    //equal to targetValue
                    left.AddOrUpdate(targetValue, 1, (key, oldValue) => oldValue+1);
                }            
                else //Value is >=
                    right.AddOrUpdate(targetValue, 1, (key, oldValue) => oldValue+1);
            }

            double leftSum=0, rightSum=0;
            foreach (int value in left.Values)
                leftSum += value; //Total items from all classes
            foreach (int value in right.Values)
                rightSum += value; //Total items from all classes

            ConcurrentDictionary<double, double> sumOfLeftRight = 
                new ConcurrentDictionary<double, double>();

            //Compute impurity
            //foreach (double classValue in allClasses.Keys)
            Parallel.ForEach(allClasses.Keys, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },(classValue) =>
             {
                 double leftProportion = 0, rightProportion = 0;
                 double proportion = 0;
                 if (left.ContainsKey(classValue))
                     leftProportion = left[classValue] / leftSum;

                 if (right.ContainsKey(classValue))
                     rightProportion = right[classValue] / rightSum;

                 proportion = leftProportion * (1.0 - leftProportion) +
                                 rightProportion * (1.0 - rightProportion);

                 sumOfLeftRight.AddOrUpdate(classValue, proportion, (key, oldValue) => oldValue); //Dont do anything if present
            });

            foreach (double value in sumOfLeftRight.Values)
                giniImpurity += value;

            return giniImpurity;
        }
        
    }
}
