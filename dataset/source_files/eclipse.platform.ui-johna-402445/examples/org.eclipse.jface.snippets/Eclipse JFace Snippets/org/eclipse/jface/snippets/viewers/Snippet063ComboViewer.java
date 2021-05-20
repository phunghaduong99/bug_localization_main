/*******************************************************************************
 * Copyright (c) 2013 Hendrik Still and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Hendrik Still<hendrik.still@gammas.de> - initial implementation, bug 417676
 *******************************************************************************/

package org.eclipse.jface.snippets.viewers;

import org.eclipse.jface.layout.GridLayoutFactory;
import org.eclipse.jface.layout.LayoutConstants;
import org.eclipse.jface.viewers.ComboViewer;
import org.eclipse.jface.viewers.IStructuredContentProvider;
import org.eclipse.jface.viewers.LabelProvider;
import org.eclipse.jface.viewers.StructuredSelection;
import org.eclipse.jface.viewers.Viewer;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Label;
import org.eclipse.swt.widgets.Shell;

/**
 * A simple ComboViewer to demonstrate usage
 *
 * @author Hendrik Still <hendrik.still@gammas.de>
 *
 */
public class Snippet063ComboViewer {
	private class MyContentProvider implements IStructuredContentProvider<MyModel,MyModel[]> {

		public void dispose() {

		}

		public void inputChanged(Viewer<? extends MyModel[]> viewer, MyModel[] oldInput,
				MyModel[] newInput) {
		}

		public MyModel[] getElements(MyModel[] inputElement) {
			return inputElement;
		}

	}

	public class MyModel {
		public int counter;

		public MyModel(int counter) {
			this.counter = counter;
		}

		@Override
		public String toString() {
			return "Item " + this.counter;
		}
	}

	public Snippet063ComboViewer(Shell shell) {

		GridLayoutFactory.fillDefaults().numColumns(2)
				.margins(LayoutConstants.getMargins()).generateLayout(shell);

		final Label l = new Label(shell, SWT.None);
		l.setText("Choose Item:");
		final ComboViewer<MyModel,MyModel[]> v = new ComboViewer<MyModel,MyModel[]>(shell);
		v.setLabelProvider(new LabelProvider<MyModel>());
		v.setContentProvider(new MyContentProvider());

		MyModel[] model = createModel();
		v.setInput(model);

		// Select the initial Element
		if (model.length > 0) {
			v.setSelection(new StructuredSelection(model[0]));
		}
	}

	private MyModel[] createModel() {
		MyModel[] elements = new MyModel[11];

		for (int i = 0; i < 10; i++) {
			elements[i] = new MyModel(i);
		}
		elements[10] = new MyModel(42);

		return elements;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Display display = new Display();
		Shell shell = new Shell(display);
		new Snippet063ComboViewer(shell);

		shell.pack();
		shell.open();

		while (!shell.isDisposed()) {
			if (!display.readAndDispatch()) {
				display.sleep();
			}
		}

		display.dispose();

	}

}
