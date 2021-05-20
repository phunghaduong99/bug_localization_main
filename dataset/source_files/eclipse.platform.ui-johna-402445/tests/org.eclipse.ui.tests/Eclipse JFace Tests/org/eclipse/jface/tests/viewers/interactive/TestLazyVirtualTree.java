/*******************************************************************************
 * Copyright (c) 2005, 2006 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.jface.tests.viewers.interactive;

import org.eclipse.jface.tests.viewers.TestElement;
import org.eclipse.jface.tests.viewers.TestModelLazyTreeContentProvider;
import org.eclipse.jface.viewers.TreeViewer;
import org.eclipse.jface.viewers.Viewer;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Event;
import org.eclipse.swt.widgets.Listener;
import org.eclipse.swt.widgets.Tree;
import org.eclipse.swt.widgets.TreeItem;

public class TestLazyVirtualTree extends TestTree {

	public Viewer<TestElement> createViewer(Composite parent) {
		Tree tree = new Tree(parent, SWT.VIRTUAL);
		tree.addListener(SWT.SetData, new Listener() {
			private String getPosition(TreeItem item) {
				TreeItem parentItem = item.getParentItem();
				if (parentItem == null) {
					return "" + item.getParent().indexOf(item);
				}
				return getPosition(parentItem) + "." + parentItem.indexOf(item);
			}

			public void handleEvent(Event event) {
				String position = getPosition((TreeItem) event.item);
				System.out.println("updating " + position);
			}
		});
		TreeViewer<TestElement,TestElement> viewer = new TreeViewer<TestElement,TestElement>(tree);
		viewer.setContentProvider(new TestModelLazyTreeContentProvider(viewer));
		viewer.setUseHashlookup(true);

		if (fViewer == null)
			fViewer = viewer;
		return viewer;
	}

	public void setInput(TestElement input) {
		if(fViewer!=null) {
			Object oldInput = fViewer.getInput();
			if(oldInput!=null) {
				fViewer.setChildCount(oldInput, 0);
			}
		}
		super.setInput(input);
		if(fViewer!=null && input!=null) {
			fViewer.setChildCount(input, input.getChildCount());
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TestBrowser browser = new TestLazyVirtualTree();
		if (args.length > 0 && args[0].equals("-twopanes"))
			browser.show2Panes();
		browser.setBlockOnOpen(true);
		browser.open(TestElement.createModel(3, 10));
	}

}
