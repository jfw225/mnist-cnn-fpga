import { useEffect, useState } from 'react';
import './assets/css/App.css';

import FetchReport from "./ReportFile";
import "./assets/css/w3.css";
import { HeaderGif } from './assets';
import { ParseMD } from './MDParser';

import hljs from "highlight.js";
import "highlight.js/styles/github-dark-dimmed.css";

function App() {
	// const [htmlFile, setHtmlFile] = useState({ __html: "<div> <h3> hi </h3> </div>" })
	const [MDFile, setMDFile] = useState("");

	useEffect(() => {
		FetchReport(setMDFile);

	}, []);

	useEffect(() => {
		document.querySelectorAll("pre code").forEach(block => {
			hljs.highlightBlock(block as HTMLElement);
		});
	})
	// console.log(MDFile);
	let k = ParseMD(MDFile);

	return (
		<body className="tBody">
			<link rel="stylesheet" href="/path/to/styles/default.css" />

			<link
				href="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css"
				rel="stylesheet"
			/>

			<div className="w3-top">
				<div className="w3-bar w3-black w3-card">
					<a className="w3-bar-item w3-button w3-padding-large w3-hide-medium w3-hide-large w3-right" href="javascript:void(0)" title="Toggle Navigation Menu"><i className="fa fa-bars"></i></a>
					<a href="#" className="w3-bar-item w3-button w3-padding-large">HOME</a>
				</div>
			</div>

			<div className="w3-content" style={{ maxWidth: 2000, marginTop: 46 }}>
				{/* <div className="tContent"> */}
				<div className="mySlides w3-display-container w3-center">
					<img src={HeaderGif} className="header-image" />

				</div>
				{/* <div className="w3-white"> */}
				{/* <div className="w3-container w3-content w3-center w3-padding-64" style={{ maxWidth: 800 }} id="report"> */}
				{/* <h2 className="w3-wide">Report</h2> */}
				{k}
				{/* <MarkdownRender>{MDFile}</MarkdownRender> */}
				{/* </div> */}
				{/* </div> */}
			</div>
		</body >

	);


	// return (
	// 	<div className="App">
	// <link
	// 	href="//cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css"
	// 	rel="stylesheet"
	// />
	// 		<header className="App-header">
	// 			<img src={logo} className="App-logo" alt="logo" />
	// 			<MarkdownRender>{MDFile}</MarkdownRender>
	// 		</header>
	// 	</div>
	// );
}

export default App;
