[@react.component]
let make = () => {
    let foot = ReactDOM.Style.make
        ( ~width="100%"
        , ~height="46px"
        , ~background="#e8e6e6"
        , ~color="black"
        , ~fontSize="14px"
        , ()
        );
    <div style={foot}>
        <Center>
            {React.string("©2020 by HongShen (ustchcl@gmail.com)")}
        </Center>
    </div>
}