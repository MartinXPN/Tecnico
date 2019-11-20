import React, {Component} from "react";
import * as d3 from "d3";

interface Props {
    width: number | string;
    height: number | string;
    data: d3.DSVParsedArray<{ country: string, year: number, gdp: number, meat_consumption: number, temperature: number, ghg_emission: number }> | undefined;
}

export default class ScatterPlot extends Component<Props> {
    // @ts-ignore
    ref: SVGSVGElement;

    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        if (!this.props.data)
            return;

        const svg = d3.select(this.ref);
        const rect = this.ref.getBoundingClientRect();

        const w = rect.width;
        const h = rect.height;
        const padding = rect.width / 10;
        const dataset = this.props.data.map(row => [row.ghg_emission, row.temperature]);

        const xScale = d3.scaleLinear()
            // @ts-ignore
            .domain([d3.min(dataset, d => d[0]), d3.max(dataset, d => d[0])])
            .range([padding, w - padding * 2]);

        const yScale = d3.scaleLinear()
            // @ts-ignore
            .domain([d3.min(dataset, d => d[1]), d3.max(dataset, d => d[1])])
            .range([h - padding, padding]);

        // @ts-ignore
        const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat((val: number, _id: number) => {return '' + Math.round(val / 1000)+ 'K'});
        const yAxis = d3.axisLeft(yScale).ticks(5);

        svg.selectAll("circle")
            .data(dataset)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(d[0]))
            .attr("cy", d => h - yScale(d[1]))
            .attr("r", 2)
            .attr("fill", "green");

        //x axis
        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (h - padding) + ")")
            .call(xAxis);

        //y axis
        svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + padding + ", 0)")
            .call(yAxis);
    }

    render(): React.ReactElement {
        return (
            <svg className="container"
                 ref={(ref: SVGSVGElement) => this.ref = ref}
                 width={this.props.width}
                 height={this.props.height}>
            </svg>
        );
    }
}
