import React, {Component} from "react";
import * as d3 from "d3";
import {GdpTemperatureMeatGhgData} from "../entities";

interface Props {
    width: number | string;
    height: number | string;
    data: d3.DSVParsedArray<GdpTemperatureMeatGhgData>;
    yearStart: number;
    yearEnd: number;
    selectedCountries: Set<string>;
}

interface State {
    countriesDisplayed: Set<string>;
}

export default class ScatterPlot extends Component<Props, State> {
    private readonly countryToData: Map<string, Map<number, GdpTemperatureMeatGhgData>>;
    // @ts-ignore
    private ref: SVGSVGElement;
    // @ts-ignore
    private xScale: d3.ScaleLinear<number, number>;
    // @ts-ignore
    private yScale: d3.ScaleLinear<number, number>;
    // @ts-ignore
    private tooltip: d3.Selection<d3.BaseType, unknown, HTMLElement, any>;

    state = {
        countriesDisplayed: new Set<string>(),
        countryToData: new Map<string, Map<number, GdpTemperatureMeatGhgData>>(),
    };

    constructor(props: Props) {
        super(props);

        this.countryToData = new Map();
        props.data.forEach(d => {
            if (!this.countryToData.has(d.country))
                this.countryToData.set(d.country, new Map());
            // @ts-ignore
            this.countryToData.get(d.country).set(d.year, d);
        });

        console.log(this.countryToData);
    }

    componentDidMount(): void {
        const svg = d3.select(this.ref);
        const rect = this.ref.getBoundingClientRect();

        const w = rect.width;
        const h = rect.height;
        const padding = 50;

        this.xScale = d3.scaleLinear()
        // @ts-ignore
            .domain([d3.min(this.props.data, d => d.ghg_emission), d3.max(this.props.data, d => d.ghg_emission)])
            .range([padding, w - padding * 2]);

        this.yScale = d3.scaleLinear()
        // @ts-ignore
            .domain([d3.min(this.props.data, d => d.temperature), d3.max(this.props.data, d => d.temperature)])
            .range([h - padding, padding]);

        // @ts-ignore
        const xAxis = d3.axisBottom(this.xScale).ticks(5).tickFormat((val: number, _id: number) => '' + Math.round(val / 1000000) + 'M');
        const yAxis = d3.axisLeft(this.yScale).ticks(5);

        //x axis
        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (h - padding) + ")")
            .call(xAxis);
        svg.append("text")
            .attr("transform", "translate(" + ((w - padding) / 2) + " ," + (h - 15) + ")")
            .style("text-anchor", "middle")
            .text("GHG Emissions")
            .attr('font-size', '12px');

        //y axis
        svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + padding + ", 0)")
            .call(yAxis);
        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 10)
            .attr("x", -h / 2)
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Temperature ℃")
            .attr('font-size', '12px');


        this.tooltip = d3.select("body")
            .append("foreignObject")
            .append("xhtml:body")
            .style("position", "absolute")
            .style("z-index", "10")
            .style("visibility", "hidden")
            .style("font", "11px 'Helvetica Neue'");

        this.componentDidUpdate(this.props, this.state);
    }

    handleCountryYear = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
                         dataPoint: GdpTemperatureMeatGhgData | undefined,
                         country: string, identifier: string,
                         color: string, h: number) => {
        if(!dataPoint) {
            svg.select(`circle[title='${identifier}-${country}']`).attr('visibility', 'hidden');
            return;
        }

        svg.select(`circle[title='${identifier}-${dataPoint.country}']`)
            .on("mouseover", () => {
                this.tooltip.style("visibility", "visible");
                this.tooltip.html(`<div><strong>${dataPoint.country}</strong></div>${Math.round(dataPoint.ghg_emission / 100000) / 10 + 'M'} greenhouse gas emissions<div>${dataPoint.temperature}℃ average yearly temperature</div>`);
            })
            .on("mousemove", () => this.tooltip.style("top", (d3.event.pageY - 10) + "px").style("left", (d3.event.pageX + 10) + "px"))
            .on("mouseout", () => this.tooltip.style("visibility", "hidden"))
            .transition().duration(250)
            .attr('cx', this.xScale(dataPoint ? dataPoint.ghg_emission : 0))
            .attr('cy', h - this.yScale(dataPoint ? dataPoint.temperature : 0))
            .attr('r', 4)
            .attr("fill", color)
            .attr('visibility', dataPoint ? 'visible' : 'hidden');
    };

    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const h = this.ref.getBoundingClientRect().height;
        const svg = d3.select(this.ref);
        console.log(this.props);

        // remove old countries
        this.state.countriesDisplayed.forEach(country => {
            if (!this.props.selectedCountries.has(country)) {
                svg.select(`circle[title='yearStart-${country}']`).remove();
                svg.select(`circle[title='yearEnd-${country}']`).remove();
            }
        });

        // add new countries and display the data
        this.props.selectedCountries.forEach(country => {
            if (!this.state.countriesDisplayed.has(country)) {
                svg.append(`circle`).attr('title', `yearStart-${country}`);
                svg.append(`circle`).attr('title', `yearEnd-${country}`);
            }

            // @ts-ignore
            const start = this.countryToData.get(country).get(this.props.yearStart);
            // @ts-ignore
            const end = this.countryToData.get(country).get(this.props.yearEnd);

            this.handleCountryYear(svg, start, country, 'yearStart', "green", h);
            this.handleCountryYear(svg, end, country, 'yearEnd', "red", h);
        });
    }

    render(): React.ReactElement {
        return (
            <svg
                ref={(ref: SVGSVGElement) => this.ref = ref}
                width={this.props.width}
                height={this.props.height} />
        );
    }
}
