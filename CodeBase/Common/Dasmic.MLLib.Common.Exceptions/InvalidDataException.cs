using System;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class InvalidDataException : Exception
    {

        public InvalidDataException(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public InvalidDataException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_invalid_data;
            }
        }
    }
}
