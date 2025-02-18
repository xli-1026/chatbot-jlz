import "./index.css";

import { useContext, useState, useRef, useEffect } from "react";

import FormControl from '@mui/material/FormControl';
import FormHelperText from '@mui/material/FormHelperText';
import Input from '@mui/material/Input';

import { UserContext, ConversationContext } from "contexts";

/**
 * @param {Object} props
 * @param {string} props.chatId
 * @param {*} props.onSend
 */
const ChatInput = (props) => {
  const username = useContext(UserContext);
  const { conversations, dispatch } = useContext(ConversationContext);

  const [input, setInput] = useState("");
  const inputRef = useRef(null);

  /**
   * Focus on input when chatId changes.
   */
  useEffect(() => {
    if (props.chatId) {
      inputRef.current.focus();
    }
  }, [props.chatId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = input;
    setInput("");
    // append user input to chatlog
    dispatch({
      type: "messageAdded",
      id: props.chatId,
      message: { from: username, content: payload },
    });
    // if current chat is not the first in the list, move it to the first when send message.
    if (conversations[0].id !== props.chatId) {
      dispatch({
        type: "moveToFirst",
        id: props.chatId,
      });
    }
    await props.onSend(props.chatId, payload);
  };

  const handleKeyDown = async (e) => {
    // TODO: this will trigger in Chinese IME on OSX
    if (e.key === "Enter") {
      if (e.ctrlKey || e.shiftKey || e.altKey) {
        // won't trigger submit here, but only shift key will add a new line
        return true;
      }
      e.preventDefault();
      await handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="input-form">
      <FormControl variant="standard" className="input-form-control">
        <Input
          id="chat-input"
          // TODO: className not working
          // className="input-text"
          inputProps={{
            style: {
              padding: "12px",
              color: "white",
              fontSize: "1.25em",
            },
          }}
          multiline
          inputRef={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          aria-describedby="input-helper-text"
        />
        <FormHelperText
          id="input-helper-text"
          // TODO: className not working
          // className="input-helper"
          sx={{
            color: "white",
            paddingLeft: "12px",
          }}
        >
          Enter to send message, Shift + Enter to add a new line
        </FormHelperText>
      </FormControl>
      <button className="input-submit-button" type="submit">
        Send
      </button>
    </form>
  );
};

export default ChatInput;
