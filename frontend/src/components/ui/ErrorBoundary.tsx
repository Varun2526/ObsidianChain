/**
 * A rendering failure in one page must not blank the whole workstation.
 * The error is shown in place, with the route still navigable.
 */
import { Component, type ReactNode } from "react";

export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="state" role="alert">
        <div className="banner banner-error">
          <h4>This view failed to render</h4>
          <p>{this.state.error.message}</p>
        </div>
        <button type="button" className="btn btn-sm" onClick={() => this.setState({ error: null })}>Try again</button>
      </div>
    );
  }
}
