// This is the ReactJS documentation's useReducer example, directly ported over
// https://reactjs.org/docs/hooks-reference.html#usereducer

// A little extra we've put, because the ReactJS example has no styling
let leftButtonStyle = ReactDOMRe.Style.make(~borderRadius="4px 0px 0px 4px", ~width="48px", ());
let centerButtonStyle = ReactDOMRe.Style.make(~width="64px", ());
let rightButtonStyle = ReactDOMRe.Style.make(~borderRadius="0px 4px 4px 0px", ~width="48px", ());
let containerStyle = ReactDOMRe.Style.make(~display="flex", ~alignItems="center", ~justifyContent="space-between", ());

// Record and variant need explicit declarations.
type state = {count: int};

type action =
  | Increment
  | Decrement
  | Set(int);

let initialState = {count: 0};

let reducer = (state, action) => {
  switch (action) {
  | Increment => {count: state.count + 1}
  | Decrement => {count: state.count - 1}
  | Set(n) => {count: n}
  };
};

[@react.component]
let make = () => {
  let (state, dispatch) = React.useReducer(reducer, initialState);

  // We can use a fragment here, but we don't, because we want to style the counter
  <div style=containerStyle>
    <div>
      {React.string("Count: ")}
      {React.string(string_of_int(state.count))}
    </div>
    <div>
      <button style=leftButtonStyle onClick={_event => dispatch(Decrement)}>
        {React.string("-")}
      </button>
      <button style=centerButtonStyle onClick={_event => dispatch(Set(20))}>
        {React.string("set 20")}
      </button>
      <button style=rightButtonStyle onClick={_event => dispatch(Increment)}>
        {React.string("+")}
      </button>
    </div>
  </div>;
};
