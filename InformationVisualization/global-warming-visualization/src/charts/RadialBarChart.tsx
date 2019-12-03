import React, {Component} from "react";
import * as d3 from "d3";
import {SeaGlaciersData as Data} from "../entities";
import Tooltip from "../tooltip/Tooltip";

interface Props {
    width: number | string;
    height: number | string;
    data: d3.DSVParsedArray<Data>;
    yearStart: number;
    yearEnd: number;
}
interface State {
    hoveredYear: number | undefined;
}

function scaleRadial(domain: number[], range: number[]) {

    const scale = (x: number) => {
        const r0 = range[0] * range[0], r1 = range[1] * range[1];
        return Math.sqrt((x - domain[0]) / (domain[1] - domain[0]) * (r1 - r0) + r0);
    };

    scale.domain = (_: number[]) => _.length ? (domain = [+_[0], +_[1]], scale) : domain.slice();
    scale.range = (_: number[]) => _.length ? (range = [+_[0], +_[1]], scale) : range.slice();
    scale.ticks = (count: number) => d3.scaleLinear().domain(domain).ticks(count);
    scale.tickFormat = (count: number, specifier: any) => d3.scaleLinear().domain(domain).tickFormat(count, specifier);
    return scale;
}

export default class RadialBarChart extends Component<Props, State> {
    private static SEA_LEVEL_COLOR = "#1484b3";
    private static GLACIER_MASS_COLOR = "#b32019";
    private static OPACITIES = {DISABLED: 0.1, ENABLED: 0.7, HIGHLIGHTED: 1};
    protected title = 'Sea level and glaciers mass';
    // @ts-ignore
    private ref: SVGSVGElement;
    private seaLevelElements: d3.Selection<SVGPathElement, Data, SVGElement, unknown> | undefined;
    private glacierElements: d3.Selection<SVGPathElement, Data, SVGElement, unknown> | undefined;
    protected tooltip: Tooltip;

    constructor(props: Props) {
        super(props);
        this.tooltip = new Tooltip({});
    }


    state = {
        hoveredYear: undefined,
    };

    addRadialChart = (elementSet: d3.Selection<SVGPathElement, Data, SVGElement, unknown>,
                      innerRadius: number, outerRadius: number,
                      color: string, range: number[],
                      data: d3.DSVParsedArray<Data>,
                      getValue: (d: any) => number, getX: (d: any) => number,
                      getDescription: (d: any) => string) => {

        const smallest = d3.min(data, getValue);
        const largest = d3.max(data, getValue);

        const x = d3.scaleBand()
            .range([range[0], range[1]])    // X axis goes from 0 to 2pi = all around the circle. If I stop at 1Pi, it will be around a half circle
            .domain(data.map(d => "" + getX(d))); // The domain of the X axis is the list of states.

        const y = scaleRadial(
            // @ts-ignore
            [smallest, largest],
            [innerRadius, outerRadius]
        );

        elementSet.on("mouseover", (d) => {
            this.tooltip.show(getDescription(d));
            this.setState({hoveredYear: d.year});
        })
            .on("mousemove", () => this.tooltip.move(d3.event.pageY - 10, d3.event.pageX + 10))
            .on("mouseout", () => {
                this.tooltip.hide();
                this.setState({hoveredYear: undefined});
            })
            .attr("fill", color)
            .attr("d", d3.arc()
                .innerRadius(innerRadius)
                .outerRadius(d => {
                    // @ts-ignore
                    const heightMultiplier = d.year === this.state.hoveredYear ? 1.1 : 1;
                    return y(heightMultiplier * getValue(d))
                })
                // @ts-ignore
                .startAngle(d => {return x("" + getX(d));}).endAngle(d => {return x("" + getX(d)) + x.bandwidth();})
                .padAngle(0.01)
                .padRadius(innerRadius)
            )
            .transition().duration(100)
            // hide or show opacity = 1 => show, opacity = 0 => hide
            .style("opacity", d => {
                if( this.state.hoveredYear === d.year )
                    return RadialBarChart.OPACITIES.HIGHLIGHTED;
                if( this.props.yearStart <= getX(d) && getX(d) <= this.props.yearEnd )
                    return RadialBarChart.OPACITIES.ENABLED;
                return RadialBarChart.OPACITIES.DISABLED;
            });
    };

    componentDidMount(): void {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        console.log(h);

        const svg = d3.select(this.ref)
            .append("g")
            .attr("transform", "translate(" + w / 2 + "," + h / 2 + ")");

        svg.append("text")
            .attr("transform", "translate(0," + (-h / 15) + ")")
            .style("text-anchor", "middle")
            .text("Sea level")
            .attr('font-size', '13px')
            .attr('font-weight', 'bold')
            .style("fill", RadialBarChart.SEA_LEVEL_COLOR);

        svg.append("text")
            .attr("transform", "translate(0," + (h / 15) + ")")
            .style("text-anchor", "middle")
            .text("Glaciers mass")
            .attr('font-size', '13px')
            .attr('font-weight', 'bold')
            .style("fill", RadialBarChart.GLACIER_MASS_COLOR);

        svg.append("text")
            .attr("transform", "translate(" + 0 + " ," + (-h / 2 + 15) + ")")
            .style("text-anchor", "middle")
            .text(this.title)
            .attr('font-weight', 'bold')
            .attr('font-size', '15px');

        // years
        const minYear = d3.min(this.props.data, d => d.year);
        const maxYear = d3.max(this.props.data, d => d.year);
        svg.append("text")
            .attr("transform", "translate(" + (-w / 6) + " ," + 0 + ")")
            .style("text-anchor", "middle")
            .text('' + minYear)
            .attr('font-size', '9px');
        svg.append("text")
            .attr("transform", "translate(" + (w / 4) + " ," + 0 + ")")
            .style("text-anchor", "middle")
            .text('' + maxYear)
            .attr('font-size', '9px');


        this.seaLevelElements = svg.append("g")
            .selectAll("path")
            .data(this.props.data)
            .enter()
            .append("path");

        this.glacierElements = svg.append("g")
            .selectAll("path")
            .data(this.props.data)
            .enter()
            .append("path");
    }

    componentDidUpdate(prevProps: Readonly<Props>, prevState: Readonly<{}>, snapshot?: any): void {
        const rect = this.ref.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        const innerRadius = Math.min(w, h) / 4;
        const outerRadius = Math.min(w, h) / 2;   // the outerRadius goes from the middle of the SVG area to the border

        const minYear = d3.min(this.props.data, d => d.year);
        if (this.seaLevelElements && this.glacierElements) {
            this.addRadialChart(this.seaLevelElements, innerRadius, outerRadius, RadialBarChart.SEA_LEVEL_COLOR, [1.5 * Math.PI, 2.5 * Math.PI], this.props.data, (d) => d.level, d => d.year, (d) => `<div style="text-align: center"><strong>Year ${d.year}</strong></div>Global sea level increased by ${d.level} since ${minYear}`);
            this.addRadialChart(this.glacierElements, innerRadius, outerRadius, RadialBarChart.GLACIER_MASS_COLOR, [-0.5 * Math.PI, -1.5 * Math.PI], this.props.data, (d) => -d.mass, d => d.year, (d) => `<div style="text-align: center"><strong>Year ${d.year}</strong></div>Global glacier mass decreased by ${-d.mass} since ${minYear}`);
        }
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
