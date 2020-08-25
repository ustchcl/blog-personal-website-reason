import ReactMarkdown from 'react-markdown'
import React from 'react'
import RemarkMathPlugin from 'remark-math';
import MathJax from 'react-mathjax';
import CodeBlock from './CodeBlock.jsx';

export default function MarkdownRender(props) {
  const newProps = {
    ...props,
    plugins: [
      RemarkMathPlugin,
    ],
    renderers: {
      ...props.renderers,
      code: CodeBlock,
      math: (props) => 
        <MathJax.Node formula={props.value} />,
      inlineMath: (props) =>
        <MathJax.Node inline formula={props.value} />
    }
  };
  return (
    <MathJax.Provider input="tex">
      <ReactMarkdown {...newProps}/>
    </MathJax.Provider>
  )
}