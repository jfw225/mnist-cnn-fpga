import ReactMarkdown from 'react-markdown';
import MathJax from 'react-mathjax';
import RemarkMathPlugin from 'remark-math';
import RehypeKatex from "rehype-katex";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { light, dark } from 'react-syntax-highlighter/dist/esm/styles/prism'

function MarkdownRender(props) {
    const newProps = {
        ...props,
        remarkPlugins: [
            RemarkMathPlugin,
        ],
        rehypePlugins: [
            RehypeKatex
        ],
        components: {
            ...props.components,
            math: (props) =>
                <MathJax.Node formula={props.value} />,
            inlineMath: (props) =>
                <MathJax.Node inline formula={props.value} />,
        }
    };
    return (
        <MathJax.Provider input="tex">
            <link
                href="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css"
                rel="stylesheet"
            />
            <ReactMarkdown {...newProps} />
        </MathJax.Provider>
    );
}

export default MarkdownRender