using System;
using System.Text;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class ModelBase : Common.MLCore.ModelBase
    {
        protected DecisionTreeNode _root;

        public DecisionTreeNode Root
        {
            set
            {
                _root = value;
            }
            get
            {
                return _root;
            }
        }

        public ModelBase(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes)
        {

        }
        
        /// <summary>
        /// Return value of a target attribute. 
        /// Main assumption is that data is in same indexes as training data
        /// </summary>
        /// <param name="data"></param>
        /// <param name="indexTargetAttribute"></param>
        /// <returns></returns>
        public override 
            double RunModelForSingleData(double[] data)
        {
            try
            {
                
                double value = double.PositiveInfinity;
                if (_root == null) return value; //Model not computed
                DecisionTreeNode dtn = _root;

                
                //Deal with missing values
                while (dtn.AttributeIndex != _origTargetAttributeIndex)
                {
                    //Children are stored with Index
                    dtn = dtn.getChildWithValue(
                        data[dtn.AttributeIndex]);
                    if (dtn == null) //Is null likely due to missing value, ignore it in this case
                    {
                        return Constants.MISSING_VALUE;
                    }
                }
                value = dtn.Value;
                return value;
            }
            catch (Exception ex)
            {
                throw new ModelRunException(ex.Message, ex);
            }
        }

        //Serialization Routine
        public override void SaveModel(string filePath)
        {
            throw new NotImplementedException();
        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {
            throw new NotImplementedException();
        }

        #region Print routines
        public virtual string getPrintedTree()
        {
            return printTree(_root);
        }

        protected string printTree(DecisionTreeNode dtn)
        {
            StringBuilder sb = new StringBuilder();
            //Print dtn and child notes
            sb.Append(dtn.AttributeName);
            if (dtn.Children == null) return sb.ToString();
            //Else Has Children
            sb.Append("-->");
            foreach (DecisionTreeNode child in dtn.Children)
            {
                sb.Append(child.AttributeName + ",");
            }
            sb.Append("\r\n");

            foreach (DecisionTreeNode child in dtn.Children)
            {
                sb.Append(printTree(child));
            }
            return sb.ToString();
        }
        #endregion
    }
}
