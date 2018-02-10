using System;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class InvalidTrainingDataException:Exception
    {

        public InvalidTrainingDataException(
            string message, 
            Exception innerException):base(message,innerException)
        {
            
        }


        public InvalidTrainingDataException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_invalid_training_data;
            }
        }

       
    }

    
}
